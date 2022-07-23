import numpy as np


def group_array(array, threshold):
    """
    Search array for adjoining coordinates within array which lie above threshold, forming a group.
    A new group is created each time a coordinate is encountered which is above the threshold and has not been visited yet.
    
    :param array: (2-D ndarray) matrix which contains groups which need to be isolated and identified.
    :param threshold: (float) value necessary to categorize coordinates in array which belong to a group.
    
    :return: list of Group objects which were identified in array.
    """
    
    groups = []

    # for simplicity and speed, a boolean reference array is created which will keep track of which positions have
    # already been visited for group membership.
    reference_array = np.zeros(array.shape, dtype=bool)

    # search an array for values above a certain limit
    for row in range(array.shape[0]):
        for column in range(array.shape[1]):
            if array[row, column] >= threshold and not reference_array[row, column]:
                # when one is encountered, and is not apart of a group, create a new group
                new_group = Group(row, column, threshold, array, reference_array)
                new_group.search_and_add()  # build the group from its initial point
                groups.append(new_group)
    return groups


class Group:
    def __init__(self, row, column, threshold, array, reference_array):
        """
        Builds a group object using bredth first search that identifies and keeps track of its members.

        :param row: (int) row of the initial point of the group, from which the group will begin its search for additional members.
        :param column: (int) column of the initial point of the group, from which the group will begin its search for additional members.
        :param threshold: (float) minimum value of a point needed to be included in the group. This is to prevent accidentaly
            including noise as a group member.
        :param array: (2-D ndarray) The array from which to search for new group members.
        :param reference_array: (2-D ndarray) A boolean array of the same size as array, which is updated when positions have been
            checked for group membership.
        """

        self._group_members = [(row, column)]
        self._threshold = threshold
        self._array = array
        self._reference_array = reference_array
        self._positions_to_add = []

    def _search(self, starting_position):
        """
        Searches the surrounding points of a given row/column pair to check whether they are part of the group using
        breadth first search.

        :param row: (int) row position in 2D array
        :param column: (int) column position in 2D array
        """

        positions = valid_neighbor_positions(starting_position, self._reference_array)
        positions_to_add = []
        while positions:
            pos = positions.pop()
            if not self._reference_array[pos[0], pos[1]] and self._array[pos[0], pos[1]] >= self._threshold:
                positions_to_add.append(pos)
                self._reference_array[pos[0], pos[1]] = True
                positions.extend(valid_neighbor_positions(pos, self._reference_array))
            else:
                self._reference_array[pos[0], pos[1]] = True
        self._add_positions_to_group(positions_to_add)


    def _add_positions_to_group(self, positions):
        """
        Appends points identified in _search to the group members
        """
        
        for pos in positions:
            if pos not in self._group_members:
                self._group_members.append(pos)
        self._positions_to_add = []

    def search_and_add(self):
        """
        API for building the group from initial point.
        """
        
        self._search(self._group_members[0])
        self._add_positions_to_group(self._positions_to_add)

    def group_members(self):
        """
        :return: (list) group member tuples (x, y)
        """
        
        return self._group_members

    def walking_median(self):
        """
        :return: (list) median for each unique column (y-coordinate) in the group.
        """
        
        group_members = np.array(self._group_members)
        indicies = np.argsort(group_members[:, 0])
        group_members = group_members[indicies]
        walking_median = []
        for t in np.unique(group_members[:, 1]):
            med = np.median(group_members[group_members[:, 1] == t, 0])
            walking_median.append((t, med))
        return walking_median


def valid_neighbor_positions(pos, reference_array):
    """
    Builds a set of possible neighboring coordinates based on pos which have not already been visited based on reference_array.
    
    :param pos: (tuple) row and column coordinate.
    :param reference_array: (boolean 2-D ndarray) used to determine maximum row and column values
                            and whether potential neighbors have already been visited.
    
    :return: (list) valid, unvisited neighbors
    """
    search_rows = [0]
    search_columns = [0]
    if 0 < pos[0] < reference_array.shape[0] - 1:
        search_rows.extend([-1, 1])
    else:
        placement = True if pos[0] > 0 else False
        search_rows.append(1 - 2 * placement)
    if 0 < pos[1] < reference_array.shape[1] - 1:
        search_columns.extend([-1, 1])
    else:
        placement = True if pos[1] > 0 else False
        search_columns.append(1 - 2 * placement)
    return [comb for comb in combinations(pos[0], pos[1], search_rows, search_columns)
            if not reference_array[comb[0], comb[1]]]


def combinations(x, y, list1, list2):
    """
    Given an x and y central coordinate, returns all adjoining, valid positions. An example of an invalid position would
    be one which sits outside of an array.
    
    :param x: (int) row
    :param y: (int) column
    :param list1: (list) contains a permutation of [-1, 0, 1] which represent the valid displacement of x within an array.
                  (ex. an invalid displacement would be one which causes x + displacement to be outside of an array)
    :param list2: (list) contains a permutation of [-1, 0, 1] which represent the valid displacement of y within an array.
                  (ex. an invalid displacement would be one which causes y + displacement to be outside of an array)
    
    :return: (list) surrounding coordinates of x and y
    """
    
    combinations = []
    for row in list1:
        combinations.append((x + row, y))
    for col in list2:
        combinations.append((x , y + col))
    return combinations


def moving_average(x, n):
    """
    Returns the moving average of an array, x, with the frame dictated by n. In context, this is used to smooth the
    walking median of a group and create a "ridge" on which simple statistics can be used to determine the presence
    of a whale call.
    """
    
    try:
        t = np.zeros((x.shape[0] - (n - 1), x.shape[1]))
        cumsum = np.cumsum(np.insert(x[:, 1], 0, 0))
        t[:, 0] = x[n - 1:, 0]
        t[:, 1] = (cumsum[n:] - cumsum[:-n]) / float(n)
        return t
    except ValueError:
        return x


def group_statistics(x):
    """
    Returns a dictionary of basic statistics of an array x. In context, this is used with the moving average of a
    group. The important statistics for a whale call are the time diff and frequency diff. The time diff determines
    whether a group lasts long enough (in the x-coordinate) to be a whale call. The frequency diff determines whether
    the group has a roughly similar shape to a call. These statistics were specifically developed for Blue Whale calls,
    but should be useful for other long, song-like calls.
    
    Statistics:
    time min - minimum x-coordinate of group
    time max - maximum x-coordinate of group
    time mid - median x-coordinate of group
    time diff - difference between time max and time mid.
    frequency min - minimum y-coordinate of group
    frequency max - maximum y-coordinate of group
    frequency min - median y-coordinate of group
    frequency diff - difference between the median of the first half of the group and the second half of the group.
    
    :param x: (2-D ndarray) x-y coordinates of the moving average of a group.
    
    :return: (dict) dictionary conaining relevant statistics.
    """
    
    stat_dict = {}
    stat_dict['time min'] = np.min(x[:, 0])
    stat_dict['time max'] = np.max(x[:, 0])
    stat_dict['time mid'] = np.median(x[:, 0])
    stat_dict['time diff'] = stat_dict['time max'] - stat_dict['time min']
    stat_dict['frequency min'] = np.min(x[:, 1])
    stat_dict['frequency max'] = np.max(x[:, 1])
    stat_dict['frequency mid'] = np.median(x[:, 1])
    stat_dict['frequency diff'] = np.median(x[:int(x.shape[0] / 2), 1]) - np.median(x[int(x.shape[0] / 2):, 1])
    return stat_dict


if __name__ == '__main__':
    # simple test
    test_array = np.array([[1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                           [1, 1, 1, 0, 0, 0, 1, 0, 0, 1]])

    print(test_array, '\n')
    groups = group_array(test_array, threshold=1, max_count=1000)
    x = np.arange(len(groups))
    group_dict = {group: num + 1 for group, num in zip(groups, x)}

    y = np.zeros(test_array.shape)
    for key, value in group_dict.items():
        group_pos = key.group_members()
        for pos in group_pos:
            y[pos[0], pos[1]] = value

    print(y)