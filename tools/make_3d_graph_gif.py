import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@dataclass
class Graph:
    x_coordinates: Iterable[float]
    y_coordinates: Iterable[float]
    z_coordinates: Iterable[float]
    title: str

    def make_3d_graph(self, only_show: bool = False, static: bool = False):
        fig = plt.figure()
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)

        def initialize():
            ax.plot(self.x_coordinates, self.y_coordinates, self.z_coordinates)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(self.title)
            # Set aspect 'equal'
            xmid = (max(self.x_coordinates) + min(self.x_coordinates)) / 2
            ymid = (max(self.y_coordinates) + min(self.y_coordinates)) / 2
            zmid = (max(self.z_coordinates) + min(self.z_coordinates)) / 2
            max_range = max(
                [
                    (max(self.x_coordinates) - min(self.x_coordinates)) / 2,
                    (max(self.y_coordinates) - min(self.y_coordinates)) / 2,
                    (max(self.z_coordinates) - min(self.z_coordinates)) / 2,
                ]
            )
            ax.set_xlim(xmid - max_range, xmid + max_range)
            ax.set_ylim(ymid - max_range, ymid + max_range)
            ax.set_zlim(zmid - max_range, zmid + max_range)
            return fig

        def animate(i: int):
            ax.view_init(elev=30, azim=3.6 * i)

        if only_show:
            initialize()
            plt.show()
        elif static:
            return initialize()
        else:
            ani = animation.FuncAnimation(
                fig, animate, init_func=initialize, frames=100, interval=100, blit=False
            )
            return ani


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make 3D image from json coordinate file."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Source json file",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        default=Path("").resolve(),
        help="The path to save figure",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="over write if gif exists already",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show via plt.show(), this require only one data.",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Save it as png. if specity this, gen static image",
    )
    args = parser.parse_args()

    if args.show:
        assert len(args.files) == 1
        for file in args.files:
            acc = Path(file).stem
            with open(file) as f:
                coor = json.load(f)
            g = Graph(
                title=acc,
                x_coordinates=[d[0] for d in coor],
                y_coordinates=[d[1] for d in coor],
                z_coordinates=[d[2] for d in coor],
            )
            g.make_3d_graph(only_show=True)
    elif args.png:
        assert len(args.files) == 1
        for file in args.files:
            acc = Path(file).stem
            with open(file) as f:
                coor = json.load(f)
            g = Graph(
                title=acc,
                x_coordinates=[d[0] for d in coor],
                y_coordinates=[d[1] for d in coor],
                z_coordinates=[d[2] for d in coor],
            )
            fig = g.make_3d_graph(static=True)
            plt.savefig(Path(args.outdir) / (acc + ".png"))
    else:
        for file in args.files:
            acc = Path(file).stem
            out = Path(args.outdir) / (acc + ".gif")
            if out.exists() and not args.overwrite:
                continue
            with open(file) as f:
                coor = json.load(f)
            g = Graph(
                title=acc,
                x_coordinates=[d[0] for d in coor],
                y_coordinates=[d[1] for d in coor],
                z_coordinates=[d[2] for d in coor],
            )
            ani = g.make_3d_graph()
            ani.save(out)
            plt.clf()
