import numpy as np
import matplotlib.pyplot as plt
import pprint

pp = pprint.PrettyPrinter(indent=2)


# constants
targets = ["2dbtv2", "blas"]
Ns = [32, 64]#, 128]
numRHSs = [1, 12, 24, 36, 48, 60]
grids = ["4.4.4.4", "4.4.8.8", "8.8.8.8", "16.16.16.16"]
numSites = {
    "4.4.4.4": 4*4*4*4,
    "4.4.8.8": 4*4*8*8,
    "8.8.8.8": 8*8*8*8,
    "16.16.16.16": 16*16*16*16
}
n_reps = 100
data_len = n_reps * len(Ns) * len(numRHSs) * len(grids) # 3600 # larger after big grid and n=128 addition
data_len_reduced = data_len // 100
n_time_slices = 4
n_targets = len(targets)
slice_labels = ["malloc", "cp_in", "op", "cp_out"]



def load():
    raw_data = dict()
    for target in targets:
        reps, Ns, numRHSs = [np.zeros(data_len, dtype=np.int32) for _ in range(3)]
        grids = np.zeros(data_len, dtype=np.dtype("<U11"))
        times = np.zeros((data_len, 4), dtype=np.int32)
        with open(target + ".out", "r") as file:
            for i, raw_line in enumerate(file.readlines()[1:]):
                line = raw_line.strip().split(",")
                reps[i] = int(line[0])
                grids[i] = line[1]
                Ns[i] = int(line[2])
                numRHSs[i] = int(line[3])
                times[i, 0] = int(line[5])
                times[i, 1] = int(line[6])
                times[i, 2] = int(line[7])
                times[i, 3] = int(line[8])
        raw_data[target] = dict()
        raw_data[target]["rep"] = np.array(reps)
        raw_data[target]["grid"] = np.array(grids)
        raw_data[target]["N"] = np.array(Ns)
        raw_data[target]["numRHS"] = np.array(numRHSs)
        raw_data[target]["time"] = np.array(times)
    return raw_data


class Plotter:
    def __init__(self, data: dict):
        self.data = data

    @staticmethod
    def get_indices_lower_median(data: np.ndarray):
        return np.argwhere(data <= np.median(data))[0 : len(data) // 2]

    def reduce(self):
        self.reduced_data = dict()
        for target in targets:
            reduced_data_target = dict()

            # parameters
            reduced_data_target["grid"] = self.data[target]["grid"][:: n_reps]
            reduced_data_target["N"] = self.data[target]["N"][:: n_reps]
            reduced_data_target["numRHS"] = self.data[target]["numRHS"][
                :: n_reps
            ]

            # reducing the times
            summed_times = np.sum(self.data[target]["time"], axis=1)
            summed_times_resized = np.resize(
                summed_times, new_shape=(data_len_reduced, n_reps)
            )

            reduced_data_target["time"] = np.zeros(
                shape=(data_len_reduced, n_time_slices),
                dtype=np.float64,
            )
            times_reshaped = np.resize(
                self.data[target]["time"],
                new_shape=(
                    data_len_reduced,
                    n_reps,
                    n_time_slices,
                ),
            )

            for i in range(data_len_reduced):
                indices = Plotter.get_indices_lower_median(summed_times_resized[i])
                time_slice = np.mean(times_reshaped[i][indices], axis=0)
                reduced_data_target["time"][i] = time_slice

            self.reduced_data[target] = reduced_data_target

    def get_reduced_time_slices(self, target: str, N: int, numRHS: int, grid: str):
        N_mask = self.reduced_data["blas"]["N"] == N
        numRHS_mask = self.reduced_data["blas"]["numRHS"] == numRHS
        grid_mask = self.reduced_data["blas"]["grid"] == grid
        index = np.argwhere(np.all([N_mask, numRHS_mask, grid_mask], axis=0))[0]
        return self.reduced_data[target]["time"][index]

    def plot_fixed_grid(self, grid: str = "8.8.8.8", normalization=None):
        fig, axs = plt.subplots(
            nrows=len(Ns),
            ncols=len(numRHSs),
            sharex=True,
            sharey=True,
            figsize=(12, 6),
        )

        units = "time in us"
        if normalization == "flops":
            units = "time in us per flop"
        elif normalization == "bandwidth":
            units =  "time in us per byte"

        cols_labels = [f"numRHS = {str(numRHS)}" for numRHS in numRHSs]
        rows_labels = [f"N = {str(N)}\n"+units for N in Ns]
        for ax, col in zip(axs[0], cols_labels):
            ax.set_title(col)
        for ax, row in zip(axs[:, 0], rows_labels):
            ax.set_ylabel(row)  # , rotation=0, size='large')

        norm_factor_func = lambda N, numRHS, grid: 1
        if normalization == "flops":
            norm_factor_func = lambda N, numRHS, grid: 8 * 9 * numRHS * N**2 * numSites[grid]

        elif normalization == "bandwidth":
            norm_factor_func = lambda N, numRHS, grid: 8 * (9*N**2 + 2 * N * numRHS) * numSites[grid]

        for N, axs_fixed_N in zip(Ns, axs):
            for numRHS, ax in zip(numRHSs, axs_fixed_N):
                slice_data = np.zeros(
                    (n_targets, n_time_slices), dtype=np.float64
                )
                for i, target in enumerate(targets):
                    slice_data[i] = self.get_reduced_time_slices(
                        target, N, numRHS, grid
                    )
                slice_data = slice_data.T / norm_factor_func(N, numRHS, grid)
                bottom = np.zeros(n_targets, dtype=np.float64)
                for i, slice_label in enumerate(slice_labels):
                    ax.bar(
                        targets, slice_data[i], label=slice_label, bottom=bottom
                    )
                    bottom += slice_data[i]
                ax.tick_params(axis="x", labelrotation=90)

        axs[0][-1].legend()

        fig.suptitle(f"BENCHMARKS: TYPE=complexF, GRID={grid}, NORM={normalization}")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    data = load()
    pp.pprint(data)
    plotter = Plotter(data)
    plotter.reduce()
    plotter.plot_fixed_grid()
    # plotter.plot_fixed_grid(normalization="flops")
    # plotter.plot_fixed_grid(normalization="bandwidth")
