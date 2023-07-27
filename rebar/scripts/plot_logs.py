import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, default="loss")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--smooth_window", type=int, default=51)
    parser.add_argument("--smooth_order", type=int, default=3)
    parser.add_argument("names_and_paths", type=str, nargs="+")
    args = parser.parse_args()

    assert len(args.names_and_paths) % 2 == 0, "must specify a name for every path"

    plt.figure()
    y_min, y_max = None, None
    for name, path in zip(args.names_and_paths[::2], args.names_and_paths[1::2]):
        data = np.load(path)[args.key]
        data = savgol_filter(data, args.smooth_window, args.smooth_order)
        s = sorted(data)
        local_y_min = s[int(len(data) * 0.01)]
        local_y_max = s[int(len(data) * 0.99)]
        if y_min is None:
            y_min, y_max = local_y_min, local_y_max
        else:
            y_min = min(y_min, local_y_min)
            y_max = min(y_max, local_y_max)
        plt.plot(data, label=name)
    plt.xlabel("step")
    plt.ylabel(args.key)
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.tight_layout()
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
