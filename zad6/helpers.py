from gamemap import GameMap


def open_map(filename: str) -> GameMap:
    state = []
    start_point = (-1, -1)
    end_point = (-1, -1)
    width = 0
    length = 0

    with open(filename, "r") as map_file:
        for index, line in enumerate(map_file):
            width = len(line)
            length += 1
            state.append([x for x in line if x != "\n"])

            if (start_point_index := line.find("S")) != -1:
                start_point = (index, start_point_index)

            if (end_point_index := line.find("E")) != -1:
                end_point = (index, end_point_index)

    return GameMap(
        start_point=start_point,
        end_point=end_point,
        state=state,
        current_point=start_point,
        width=width,
        length=length,
    )
