import numpy as np
from sklearn.cluster import KMeans


def split_lines(lines):
    kmeans = KMeans(n_clusters=2)

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1)
        angles.append([angle])

    kmeans.fit(angles)
    clusters = [[] for _ in range(kmeans.n_clusters)]

    for i, line in enumerate(lines):
        cluster_idx = kmeans.labels_[i]
        clusters[cluster_idx].append(line)

    return clusters


def _get_center_line(lines):
    start_points = np.mean([line[0][:2] for line in lines], axis=0)
    end_points = np.mean([line[0][2:] for line in lines], axis=0)

    middle_point = tuple(((start_points + end_points) / 2).astype(int))
    angle = np.arctan2(end_points[1] - start_points[1], end_points[0] - start_points[0])

    length = 300
    x1 = middle_point[0] - int(length * np.cos(angle))
    y1 = middle_point[1] - int(length * np.sin(angle))
    x2 = middle_point[0] + int(length * np.cos(angle))
    y2 = middle_point[1] + int(length * np.sin(angle))

    return [x1, y1, x2, y2]


def get_center_lines(splitted_lines):
    center_lines = []
    for cluster in splitted_lines:
        center_lines.append(_get_center_line(cluster))
    return center_lines


def get_intersection(lines):
    assert len(lines) == 2

    line1, line2 = lines

    x1_1, y1_1, x1_2, y1_2 = line1
    x2_1, y2_1, x2_2, y2_2 = line2

    m1 = (y1_2 - y1_1) / (x1_2 - x1_1) if (x1_2 - x1_1) != 0 else float("inf")
    m2 = (y2_2 - y2_1) / (x2_2 - x2_1) if (x2_2 - x2_1) != 0 else float("inf")

    if m1 == m2:
        return None

    b1 = y1_1 - m1 * x1_1
    b2 = y2_1 - m2 * x2_1

    if m1 == float("inf"):
        x = x1_1
        y = m2 * x + b2
    elif m2 == float("inf"):
        x = x2_1
        y = m1 * x + b1
    else:
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1

    return np.asarray([x, y], dtype=int)
