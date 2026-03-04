__all__ = [
    "write_smirnoff",
    "train_test_split_ds",
    "train_test_split_results",
    "plot_torsions",
    "plot_torsion_cdfs",
    "plot_torsion_rms_stats",
    "plot_torsion_mean_error_distribution",
    "plot_torsion_rms_js_distance",
]

import warnings
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
import logging

import numpy as np
from matplotlib import pyplot
from yammbs.torsion import TorsionStore
from yammbs.torsion.outputs import MetricCollection
import datasets
import descent
import descent.train
import smee
from openff.qcsubmit._pydantic import root_validator
from openff.qcsubmit.results.filters import (
    CMILESResultFilter,
)
from openff.qcsubmit.results.results import (
    BasicResultCollection,
    OptimizationResultCollection,
    TorsionDriveResultCollection,
    _BaseResult,
)
from openff.toolkit import ForceField, Molecule, Quantity
from openff.toolkit.typing.engines.smirnoff.parameters import ParameterHandler
from rdkit.Chem import AllChem, Draw, MolFromSmiles, rdFingerprintGenerator
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from tqdm.notebook import tqdm
from yammbs import MoleculeStore

HANDLERS_WITHOUT_XML_PARAMETERS = {
    "NAGLChargesHandler",
    "ToolkitAM1BCCHandler",
}

LOGGER = logging.getLogger(__name__)

def flatten[T](iterable: Iterable[Iterable[T]]) -> Iterator[T]:
    for inner in iterable:
        for item in inner:
            yield item


def train_test_split_ds(
    ds: datasets.Dataset,
    frac_test: float,
    seed: int | None = None,
) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Split ds into train and test subsets with diverse molecules in test

    Returns
    =======
    (train_ds, test_ds)
        train and test datasets
    """
    assert 0.0 < frac_test < 1.0

    n_smiles = len(ds["smiles"])
    n_test = round(frac_test * n_smiles)

    test_indices = choose_diverse_molecules(n_test, ds["smiles"], seed=seed)
    test_indices_set = test_indices
    assert len(test_indices) == len(test_indices_set)
    train_indices = [i for i in range(n_smiles) if i not in test_indices_set]

    train_ds = ds.select(indices=train_indices)
    test_ds = ds.select(indices=test_indices)
    assert len(train_ds) + len(test_ds) == len(ds)
    LOGGER.info(
        f"split dataset of {len(ds)} molecules into"
        + f" training set of {len(train_ds)} and testing set of {len(test_ds)}",
    )
    return train_ds, test_ds


class CmilesFilter(CMILESResultFilter):
    """Simple filter that compares the CMILES exactly without converting to INCHI"""

    include_cmiles: list[str] | None = None
    exclude_cmiles: list[str] | None = None

    def _filter_function(self, result: "_BaseResult") -> bool:
        return result.cmiles in self.include_cmiles

    @root_validator
    def _validate_mutually_exclusive(cls, values):
        include_cmiles = values.get("include_cmiles")
        exclude_cmiles = values.get("exclude_cmiles")

        message = (
            "exactly one of `include_cmiles` and `exclude_cmiles` must be specified"
        )

        assert include_cmiles is not None or exclude_cmiles is not None, message
        assert include_cmiles is None or exclude_cmiles is None, message

        return values


class IdFilter(CMILESResultFilter):
    """Simple filter that compares the CMILES exactly without converting to INCHI"""

    include_ids: list[int] | None = None
    exclude_ids: list[int] | None = None

    def _filter_function(self, result: "_BaseResult") -> bool:
        if self.include_ids is not None:
            return result.record_id in self.include_ids
        if self.exclude_ids is not None:
            return result.record_id not in self.exclude_ids
        raise ValueError("exactly one of `include_ids` and `exclude_ids` must be specified")

    @root_validator
    def _validate_mutually_exclusive(cls, values):
        include_ids = values.get("include_ids")
        exclude_ids = values.get("exclude_ids")

        message = "exactly one of `include_ids` and `exclude_ids` must be specified"

        assert include_ids is not None or exclude_ids is not None, message
        assert include_ids is None or exclude_ids is None, message

        return values


def train_test_split_results(
    results: (
        OptimizationResultCollection
        | TorsionDriveResultCollection
        | BasicResultCollection
    ),
    frac_test: float,
    seed: int | None = None,
) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Split ds into train and test subsets with diverse molecules in test

    Returns
    =======
    (train_results, test_results)
        train and test datasets
    """
    assert 0.0 < frac_test < 1.0

    # Sort so that the seed consistently refers to the same molecule
    smiles = sorted({result.cmiles for result in flatten(results.entries.values())})

    n_results = len(smiles)
    n_test = round(frac_test * n_results)

    test_indices = choose_diverse_molecules(n_test, smiles, seed=seed)
    test_indices_set = set(test_indices)
    assert len(test_indices) == len(test_indices_set)
    train_indices = [i for i in range(n_results) if i not in test_indices_set]

    print(
        f"{len(test_indices)} CMILES chosen for test, {len(train_indices)} chosen for train",
    )

    train_cmiles = [smiles[i] for i in train_indices]
    test_cmiles = [smiles[i] for i in test_indices]

    assert len(set(train_cmiles).intersection(test_cmiles)) == 0

    train_results = results.filter(CmilesFilter(include_cmiles=train_cmiles))
    test_results = results.filter(CmilesFilter(include_cmiles=test_cmiles))
    assert train_results.n_results + test_results.n_results == results.n_results, (
        f"{train_results.n_results=} + {test_results.n_results=} != {results.n_results=}"
    )
    print(
        f"split dataset of {results.n_results} results into"
        + f" training set of {train_results.n_results} results and testing set of {test_results.n_results}",
    )
    return train_results, test_results


def choose_diverse_molecules(
    n: int,
    smiles: Sequence[str],
    seed: int | None = None,
) -> Sequence[int]:
    """Choose n diverse molecules from a sequence of SMILES

    Returns
    =======
    indices
        The indices into the original SMILES sequence of the chosen molecules.
    """
    fingerprinter = rdFingerprintGenerator.GetMorganGenerator(radius=3)
    fingerprints = [
        fingerprinter.GetFingerprint(MolFromSmiles(s))
        for s in tqdm(
            smiles,
            desc="Computing fingerprints",
            ncols=80,
        )
    ]

    picker = MaxMinPicker()
    return picker.LazyBitVectorPick(
        fingerprints,
        len(fingerprints),
        n,
        seed=-1 if seed is None else seed,
    )


def get_torsion_image(torsion_id: int, store: TorsionStore) -> pyplot.Figure:
    """Plot the torsion image for a given molecule ID."""
    smiles = store.get_smiles_by_torsion_id(torsion_id)
    dihedral_indices = store.get_dihedral_indices_by_torsion_id(torsion_id)

    # Use the mapped SMILES to get the molecule
    mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
    if mol is None:
        raise ValueError(f"Could not convert SMILES to molecule: {smiles}")

    rdmol = mol.to_rdkit()

    # Draw in 2D - compute 2D coordinates
    AllChem.Compute2DCoords(rdmol)
    # Highlight the dihedral
    atom_indices = [
        dihedral_indices[0],
        dihedral_indices[1],
        dihedral_indices[2],
        dihedral_indices[3],
    ]
    bond_indices = [
        rdmol.GetBondBetweenAtoms(atom_indices[0], atom_indices[1]).GetIdx(),
        rdmol.GetBondBetweenAtoms(atom_indices[1], atom_indices[2]).GetIdx(),
        rdmol.GetBondBetweenAtoms(atom_indices[2], atom_indices[3]).GetIdx(),
    ]
    img = Draw.MolToImage(
        rdmol,
        size=(300, 300),
        kekulize=True,
        wedgeBonds=True,
        highlightAtoms=atom_indices,
        highlightBonds=bond_indices,
    )
    # img = Draw.MolToImage(rdmol, size=(300, 300), kekulize=True, wedgeBonds=True)

    # Return the image so that it can be added to a matplotlib figure
    return img


def plot_torsions(plot_dir: str, force_fields: list[str], store: TorsionStore) -> None:
    """Plot the torsional energies for each molecule in the dataset."""
    n_rows = 8
    n_cols = 5

    # Adjust number of rows and columns down if we have fewer than 40 molecules
    n_molecules = len(store.get_torsion_ids())
    if n_molecules * 2 < n_rows * n_cols:
        n_rows = n_molecules // n_cols
        if n_molecules % n_cols != 0:
            n_rows += 1
    n_rows *= 2  # Two rows for each molecule

    n_torsions = n_rows * n_cols / 2  # Half the axes are for images

    fig, axes = pyplot.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))

    for i, torsion_id in enumerate(store.get_torsion_ids()):
        # Draw the molecule on the upper axis and the torsion plot on the lower axis
        if i >= n_torsions:
            break

        # Put the image on upper rows and the torsion plots underneath
        col = i % n_cols
        row = i // n_cols
        image_axis = axes[row * 2, col]
        torsion_axis = axes[row * 2 + 1, col]

        # Draw the molecule
        image_axis.imshow(get_torsion_image(torsion_id, store))
        image_axis.axis("off")

        # Plot the torsion data
        torsion_axis.set_title(f"ID: {torsion_id}")
        _qm = store.get_qm_energies_by_torsion_id(torsion_id)

        _qm = dict(sorted(_qm.items()))

        qm_minimum_index = min(_qm, key=_qm.get)

        # Make a new dict to avoid in-place modification while iterating
        qm = {key: _qm[key] - _qm[qm_minimum_index] for key in _qm}

        # Assume a default grid spacing of 15 degrees (BespokeFit default)
        angles = np.arange(-165, 195, 15)
        assert len(angles) == len(qm), "QM data and angles should match in length"

        torsion_axis.plot(
            angles,
            qm.values(),
            "k.-",
            label="QM",
        )

        for force_field in force_fields:
            mm = dict(
                sorted(
                    store.get_mm_energies_by_torsion_id(
                        torsion_id,
                        force_field=force_field,
                    ).items(),
                ),
            )
            if len(mm) == 0:
                continue

            torsion_axis.plot(
                angles,
                [val - mm[qm_minimum_index] for val in mm.values()],
                "o--",
                label=force_field,
            )

        # Only add the axis if this is the last in the row - and add it off to the right
        if col == n_cols - 1:
            torsion_axis.legend(loc=0, bbox_to_anchor=(1.05, 1), borderaxespad=0)

        # Label the axes
        torsion_axis.set_ylabel(r"Energy / kcal mol$^{-1}$")
        torsion_axis.set_xlabel("Torsion angle / degrees")

    # Hide any unused axes
    for i in range(n_molecules, n_rows * n_cols):
        if i >= n_torsions:
            break
        col = i % n_cols
        row = i // n_cols
        axes[row * 2, col].axis("off")
        axes[row * 2 + 1, col].axis("off")

    fig.tight_layout()
    Path(plot_dir).mkdir(exist_ok=True, parents=True)
    fig.savefig(f"{plot_dir}/torsions.png", dpi=300, bbox_inches="tight")


def update_parameters(
    handler: ParameterHandler,
    potential: smee.TensorPotential,
    config: descent.train.ParameterConfig | None,
):

    for key, values in zip(
        potential.parameter_keys,
        potential.parameters,
        strict=True,
    ):
        if key.associated_handler in HANDLERS_WITHOUT_XML_PARAMETERS:
            continue
        parameter = handler[key.id]
        for name, unit, value in zip(
            potential.parameter_cols,
            potential.parameter_units,
            values,
            strict=True,
        ):
            if config is not None and name not in config.cols:
                continue
            name = name if key.mult is None else f"{name}{key.mult + 1}"
            try:
                setattr(parameter, name, Quantity(value, unit))
            except Exception:
                warnings.warn(
                    f"    COULD NOT UPDATE {key.id=} {name=} {unit=} {value=} {key.mult=}",
                )


def update_attributes(
    handler: ParameterHandler,
    potential: smee.TensorPotential,
    config: descent.train.AttributeConfig | None,
):
    for name, value, unit in zip(
        [] if potential.attribute_cols is None else potential.attribute_cols,
        [] if potential.attributes is None else potential.attributes,
        [] if potential.attribute_units is None else potential.attribute_units,
        strict=True,
    ):
        if config is not None and name not in config.cols:
            continue
        setattr(handler, name, Quantity(value, unit))


def write_smirnoff(
    initial_ff: ForceField,
    optimized_tensor_ff: smee.TensorForceField,
    parameters: None | dict[str, descent.train.ParameterConfig] = None,
    attributes: None | dict[str, descent.train.AttributeConfig] = None,
):

    optimized_smirnoff_ff = ForceField(initial_ff.to_string())
    for potential in optimized_tensor_ff.potentials:
        print(potential.type)
        handler = optimized_smirnoff_ff[potential.type]
        if parameters is None or potential.type in parameters:
            print("  updating parameters")
            update_parameters(
                handler,
                potential,
                None if parameters is None else parameters[potential.type],
            )
        if attributes is None or potential.type in attributes:
            print("  updating attributes")
            update_attributes(
                handler,
                potential,
                None if attributes is None else attributes[potential.type],
            )
    return optimized_smirnoff_ff


def plot_metrics(store: MoleculeStore, force_fields: list[str], plot_dir: str|Path):
    """Plot metrics of a list of force fields."""
    metrics = store.get_metrics()

    x_ranges = {
        "dde": (-16.0, 16.0),
        "rmsd": (-0.3, 3.3),
        "tfd": (-0.05, 0.55),
    }

    # metrics are stored with force field at the top of the hierarchy,
    # restructure it so that the type of metric is at the top

    # these each keep the qcarchive_id in case it's useful, though they're not
    # used in this script
    ddes: dict[str, dict[int, float | None]] = {
        force_field: {key: val.dde for key, val in metrics.metrics[force_field].items()}
        for force_field in metrics.metrics.keys()
    }

    rmsds = {
        force_field: {
            key: val.rmsd for key, val in metrics.metrics[force_field].items()
        }
        for force_field in metrics.metrics.keys()
    }

    tfds = {
        force_field: {key: val.tfd for key, val in metrics.metrics[force_field].items()}
        for force_field in metrics.metrics.keys()
    }

    data = {
        "dde": ddes,
        "rmsd": rmsds,
        "tfd": tfds,
    }
    for key in ["dde", "rmsd", "tfd"]:
        figure, axis = pyplot.subplots()

        for force_field in force_fields:
            if key == "dde":
                _data = np.array(
                    [*data[key][force_field].values()],
                    dtype=float,
                )

                counts, bins = np.histogram(
                    _data[np.isfinite(_data)],
                    bins=np.linspace(-15, 15, 31),
                )

                axis.stairs(counts, bins, label=force_field)

                axis.set_ylabel("Count")

            else:
                sorted_data = np.sort([*data[key][force_field].values()])

                axis.plot(
                    sorted_data,
                    np.arange(1, len(sorted_data) + 1) / len(sorted_data),
                    "-",
                    label=force_field,
                )

                axis.set_ylabel("CDF")

                axis.set_xlim(x_ranges[key])
                axis.set_ylim((-0.05, 1.05))
            axis.set_xlabel(key)

        axis.legend(loc=0)

        figure.savefig(Path(plot_dir) / f"{key}.png", dpi=300)


def plot_torsion_cdfs(force_fields: list[str], metrics: MetricCollection, plot_dir: str):
    """Plot the cumulative distribution functions for the RMSD, RMSE, and Jensen-Shannon distance."""
    x_ranges = {"rmsd": (0, 0.14), "rmse": (-0.3, 5), "js_distance": (None, None)}

    units = {
        "rmsd": r"$\mathrm{\AA}$",
        "rmse": r"kcal mol$^{-1}$",
        "js_distance": "",
    }

    rmsds = {
        force_field: {
            key: val.rmsd for key, val in metrics.metrics[force_field].items()
        }
        for force_field in metrics.metrics.keys()
    }

    rmses = {
        force_field: {
            key: val.rmse for key, val in metrics.metrics[force_field].items()
        }
        for force_field in metrics.metrics.keys()
    }

    js_dists = {
        force_field: {
            key: val.js_distance[0] for key, val in metrics.metrics[force_field].items()
        }
        for force_field in metrics.metrics.keys()
    }

    js_div_temp = list(list(metrics.metrics.values())[0].values())[0].js_distance[1]

    data = {
        "rmsd": rmsds,
        "rmse": rmses,
        "js_distance": js_dists,
    }
    for key in ["rmsd", "rmse", "js_distance"]:
        figure, axis = pyplot.subplots()

        for force_field in force_fields:
            if key == "dde":
                _data = np.array(
                    [*data[key][force_field].values()],
                    dtype=float,
                )

                counts, bins = np.histogram(
                    _data[np.isfinite(_data)],
                    bins=np.linspace(-15, 15, 31),
                )

                axis.stairs(counts, bins, label=force_field)

                axis.set_ylabel("Count")

            else:
                sorted_data = np.sort([*data[key][force_field].values()])

                axis.plot(
                    sorted_data,
                    np.arange(1, len(sorted_data) + 1) / len(sorted_data),
                    "-",
                    label=force_field,
                )

                x_label = (
                    key.upper() + " / " + units[key]
                    if key != "js_distance"
                    else f"Jensen-Shannon Distance at {js_div_temp} K"
                )
                axis.set_xlabel(x_label)
                axis.set_ylabel("CDF")

                axis.set_xlim(x_ranges[key])
                axis.set_ylim((-0.05, 1.05))

        axis.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        figure.savefig(f"{plot_dir}/{key}.png", dpi=300, bbox_inches="tight")


def get_rms(array: np.ndarray) -> float:
    """Calculate the root mean square of an array."""
    return np.sqrt(np.mean(array**2))


def plot_torsion_rms_stats(
    force_fields: list[str],
    metrics: MetricCollection,
    plot_dir: str,
) -> None:
    """Plot the RMS values for the RMSD and RMSE."""

    units = {
        "rmsd": r"$\mathrm{\AA}$",
        "rmse": r"kcal mol$^{-1}$",
    }

    rms_rmses = {
        force_field: get_rms(
            np.array([val.rmse for val in metrics.metrics[force_field].values()])
        )
        for force_field in force_fields
    }

    rms_rmsds = {
        force_field: get_rms(
            np.array([val.rmsd for val in metrics.metrics[force_field].values()])
        )
        for force_field in force_fields
    }

    # Plot RMS values
    for key, data in zip(["rmsd", "rmse"], [rms_rmsds, rms_rmses]):
        figure, axis = pyplot.subplots()

        # Use different colors for each bar - the same as for the CDFs
        axis.bar(data.keys(), data.values(), color=pyplot.cm.tab10.colors)
        axis.set_ylabel(key.upper() + " / " + units[key])

        # Set x-ticks to be vertical
        pyplot.xticks(rotation=90)

        # Save the figure
        figure.tight_layout()
        figure.savefig(f"{plot_dir}/{key}_rms.png", dpi=300, bbox_inches="tight")


def plot_torsion_rms_js_distance(
    force_fields: list[str],
    metrics: MetricCollection,
    plot_dir: str,
) -> None:
    """Plot the RMS JS distance for each force field."""

    rms_js_distance = {
        force_field: get_rms(
            np.array(
                [val.js_distance[0] for val in metrics.metrics[force_field].values()]
            )
        )
        for force_field in force_fields
    }

    js_div_temp = list(list(metrics.metrics.values())[0].values())[0].js_distance[1]

    # Plot mean JS distance
    figure, axis = pyplot.subplots()

    axis.bar(
        rms_js_distance.keys(), rms_js_distance.values(), color=pyplot.cm.tab10.colors
    )
    axis.set_ylabel(f"Mean Jensen-Shannon Distance at {js_div_temp} K")

    # Set x-ticks to be vertical
    pyplot.xticks(rotation=90)

    # Save the figure
    figure.tight_layout()
    figure.savefig(f"{plot_dir}/mean_js_distance.png", dpi=300, bbox_inches="tight")


def plot_torsion_mean_error_distribution(
    force_fields: list[str],
    metrics: MetricCollection,
    plot_dir: str,
) -> None:
    """Plot the distribution of mean errors for each force field."""

    units = {
        "mean_error": r"kcal mol$^{-1}$",
    }

    mean_errors = {
        force_field: np.array(
            [val.mean_error for val in metrics.metrics[force_field].values()]
        )
        for force_field in force_fields
    }
    # Plot mean error distribution using kernel density estimation
    figure, axis = pyplot.subplots(figsize=(10, 4))
    import seaborn as sns

    for force_field in mean_errors.keys():
        sns.kdeplot(
            data=mean_errors[force_field],
            label=force_field,
            ax=axis,
        )
    axis.set_xlabel("Mean Error / " + units["mean_error"])
    axis.set_ylabel("Density")
    axis.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Save the figure
    figure.tight_layout()
    figure.savefig(
        f"{plot_dir}/mean_error_distribution.png", dpi=300, bbox_inches="tight"
    )
