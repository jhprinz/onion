################################################################################
# Visualization stuff
################################################################################

from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    Notes
    -----
    Found at stackoverflow at http://stackoverflow.com/questions/20515554/colorize-voronoi-diagram

    Was adapted to return also outward facing ridges
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    new_outers = []

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        outer = []

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            outer.append(len(new_vertices))

            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        sort_idx = np.argsort(angles)
        new_region = np.array(new_region)[sort_idx].tolist()
        new_outers.append(filter(lambda x : x in outer, new_region))

        # finish
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices), new_outers


def voronoi_plot_2D(
    centers,
    vals,
    traj=None,
    max_mfpt=0,
    outer_length=10.0,
    radius=0.5,
    color_unlimited=True,
    show_centers=False
):
    """
    Make a 2D voronoi plot

    Parameters
    ----------
    centers : numpy.ndarray, shape = (N, 2)
    vals : numpy.ndarray, shape = (N, )
    traj : numpy.ndarray, shape = (N, 2)
    max_mfpt : float
    outer_length : float
    radius : float
    color_unlimited : bool
    show_centers : bool

    Notes
    -----
    Replaces the scipy.spatial.voronoi_plot_2d function

    """
    vor = Voronoi(centers)

    fig, ax = plt.subplots()
    patches = []
    values = []
    k = len(centers)

    if traj is not None:
        plot_traj = plt.plot(traj[:,0], traj[:,1])
        plt.setp(plot_traj, color='k', linewidth=0.5, alpha=0.7)

    assignment = [None] * (max(vor.point_region) + 1)
    for n, v in enumerate(vor.point_region):
        assignment[v] = n

    val = np.zeros(k + 1)

    for n in range(k):
        val[vor.point_region[n]] = vals[n]

    regions, vertices, outers = voronoi_finite_polygons_2d(vor, radius)

    for n in range(len(regions)):
        reg = regions[n]
        if color_unlimited or -1 not in vor.regions[vor.point_region[n]]:
            if len(reg) > 0 and (max_mfpt == 0 or vals[n] < max_mfpt):
                polygon = Polygon(vertices[reg], True)
                patches.append(polygon)
                values.append(vals[n])

    center = vor.points.mean(axis=0)

    for n in range(len(vor.ridge_vertices)):
        vs = vor.ridge_vertices[n]
        ps = vor.ridge_points[n]
        if -1 in vs:
            ii = vs[1 - vs.index(-1)]
            p1 = ps[0]
            p2 = ps[1]

            v2 = ii

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * outer_length

            ppp = np.array([vor.vertices[v2],far_point])
            ax.add_line(
                plt.Line2D(
                    ppp[:,0], ppp[:,1],
                    linewidth=2,
                    color='black',
                    linestyle='dotted'
                )
            )

    if show_centers:
        plt.plot(centers[:,0], centers[:,1], 'k.')

    if color_unlimited:
        for n in range(len(outers)):
            ppp = vertices[outers[n]]
            ax.add_line(
                plt.Line2D(
                    ppp[:,0], ppp[:,1],
                    linewidth=2, color='gray',
                    linestyle='dashed'
                )
            )

    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.6)

    colors = np.array(values) / np.max(values)
    p.set_array(np.array(colors))

    ax.add_collection(p)
    ax.set_aspect('equal')