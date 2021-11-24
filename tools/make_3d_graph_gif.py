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

    def make_3d_graph(self):
        fig = plt.figure()
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)

        def initialize():
            ax.plot(self.x_coordinates, self.y_coordinates, self.z_coordinates)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(self.title)
            return fig

        def animate(i: int):
            ax.view_init(elev=30, azim=3.6 * i)

        ani = animation.FuncAnimation(
            fig, animate, init_func=initialize, frames=100, interval=100, blit=False
        )
        return ani


# def make_3d_graph(
#     graph: Graph,
# ) -> None:
#     fig = plt.figure()
#     ax = Axes3D(fig)
#
#     def initialize():
#         ax.plot(graph.x_coordinates, graph.y_coordinates, graph.z_coordinates)
#         ax.set_title(graph.title)
#         ax.set_xlabel("X")
#         ax.set_ylabel("Y")
#         ax.set_zlabel("Z")
#         return fig
#
#     def animate(i: int):
#         ax.view_init(elev=30, azim=3.6 * i)
#
#     ani = animation.FuncAnimation(
#         fig, animate, init_func=initialize, frames=100, interval=100, blit=False
#     )
#     return ani
#

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make 3D image from json coordinate file."
    )
    parser.add_argument("files", nargs="+", help="Source json file")
    parser.add_argument("--outdir", "-o", default=Path("").resolve(), help="Outdir")
    args = parser.parse_args()

    for file in args.files:
        with open(file) as f:
            coor = json.load(f)
        acc = Path(file).stem
        g = Graph(
            title=acc,
            x_coordinates=[d[0] for d in coor],
            y_coordinates=[d[1] for d in coor],
            z_coordinates=[d[2] for d in coor],
        )
        out = Path(args.outdir) / (acc + ".gif")
        ani = g.make_3d_graph()
        ani.save(out)
