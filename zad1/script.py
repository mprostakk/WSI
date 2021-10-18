import abc
import logging
from datetime import datetime
from typing import List, Optional, Tuple

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy
import numpy as np

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

Vector2D = Tuple[float, float]


def f(point: Vector2D) -> float:
    # https://en.wikipedia.org/wiki/Himmelblau%27s_function
    x, y = point[0], point[1]
    return (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2


def f_dx(point: Vector2D) -> float:
    x, y = point[0], point[1]
    return 4 * (x ** 3) + 4 * x * y - 42 * x + 2 * (y ** 2) - 14


def f_dy(point: Vector2D) -> float:
    x, y = point[0], point[1]
    return 2 * y + 2 * (x ** 2) - 22 + 4 * (y ** 3) + 4 * x * y - 28 * y


def f_dxx(point: Vector2D) -> float:
    x, y = point[0], point[1]
    return 12 * (x ** 2) + 4 * y - 42


def f_dyy(point: Vector2D) -> float:
    x, y = point[0], point[1]
    return 2 + 12 * (y ** 2) + 4 * x - 28


def f_dxy(point: Vector2D) -> float:
    x, y = point[0], point[1]
    return 4 * x + 4 * y


def f_dyx(point: Vector2D) -> float:
    return f_dxy(point)


def f_hessian(point: Vector2D) -> np.ndarray:
    return np.array(
        [
            [f_dxx(point), f_dxy(point)],
            [f_dxy(point), f_dyy(point)],
        ]
    )


class GradientDescent:
    def __init__(
        self,
        epsilon: float = 1e-12,
        iterations: int = 100,
        beta: float = 0.01,
        max_run_time_in_microseconds: int = 10000,
    ) -> None:
        self._epsilon = epsilon
        self._iterations = iterations
        self._current_iteration = 0
        self._beta = beta
        self._history_points: List[Vector2D] = list()
        self._history_time_in_microseconds: List[float] = list()
        self._start_run_datetime: Optional[datetime] = None
        self._max_run_time_in_microseconds = max_run_time_in_microseconds

    @abc.abstractmethod
    def gradient(self, point: Vector2D) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def generate_new_point(self, point: Vector2D) -> Vector2D:
        raise NotImplementedError()

    def start_point(self) -> Vector2D:
        return self._history_points[0]

    def run(self, start_point: Vector2D):
        self.add_point_to_history(start_point)
        self._start_run_datetime = datetime.now()
        point = start_point

        while not self.check_for_stop():
            start_time = datetime.now()
            new_point = self.generate_new_point(point)
            end_time = datetime.now()
            self._history_time_in_microseconds.append((end_time - start_time).microseconds)

            if (
                np.abs(point[0] - new_point[0]) < self._epsilon
                or np.abs(point[1] - new_point[1]) < self._epsilon
            ):
                logging.info("Epsilon!")
                break

            logging.info(f"New point found: {new_point[0]}, {new_point[1]}. Value: {f(new_point)}")

            self.add_point_to_history(new_point)
            self.next_iter()
            point = new_point

    def check_for_stop(self) -> bool:
        if self._current_iteration > self._iterations:
            logging.info(f"Stopped at iteration {self._iterations}")
            return True

        if (
            datetime.now() - self._start_run_datetime
        ).microseconds > self._max_run_time_in_microseconds:
            logging.info(f"Stopped after {self._max_run_time_in_microseconds} microseconds")
            return True

        return False

    def next_iter(self) -> None:
        self._current_iteration += 1

    def add_point_to_history(self, point: Vector2D):
        self._history_points.append(point)


class SteepestGradientDescent(GradientDescent):
    def gradient(self, point: Vector2D) -> Vector2D:
        return -f_dx(point), -f_dy(point)

    def generate_new_point(self, point: Vector2D) -> Vector2D:
        gradient = self.gradient(point)
        new_point = (point[0] + self._beta * gradient[0], point[1] + self._beta * gradient[1])
        return new_point


class NewtonGradientDescent(GradientDescent):
    def gradient(self, point: Vector2D) -> Vector2D:
        return -f_dx(point), -f_dy(point)

    def generate_new_point(self, point: Vector2D) -> Vector2D:
        gradient = self.gradient(point)
        gradient_vector = np.array([gradient[0], gradient[1]])

        hessian = f_hessian(point)
        hessian_inv = numpy.linalg.inv(hessian)

        d = hessian_inv.dot(gradient_vector)

        logging.info(f"{gradient[0]}, {gradient[1]}")

        new_point = (point[0] + self._beta * d[0], point[1] + self._beta * d[1])
        return new_point


class GradientPlotter:
    def __init__(self, gradient_descent: GradientDescent, fig, ax) -> None:
        self._gradient_descent = gradient_descent
        self.fig = fig
        self.ax = ax

    def plot(self):
        y, x = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
        z = f((x, y))

        self.add_title()
        self.add_heatmap(x, y, z)
        self.add_history_points()
        self.add_contours(x, y, z)

        self.ax.colorbar()

        self.ax.plot()

    def add_title(self):
        start_point = self._gradient_descent.start_point()
        title = f"Point: [{start_point[0]}, {start_point[1]}] / Beta: {self._gradient_descent._beta}"
        self.ax.set_title(title)

    def add_heatmap(self, x, y, z):
        c = self.ax.pcolor(
            x,
            y,
            z,
            norm=colors.SymLogNorm(linthresh=1, linscale=0.1, vmin=z.min(), vmax=z.max(), base=10),
            cmap="viridis_r",
        )

        self.fig.colorbar(c, ax=self.ax)

    def add_history_points(self):
        x_points = [p[0] for p in self._gradient_descent._history_points]
        y_points = [p[1] for p in self._gradient_descent._history_points]
        self.ax.plot(x_points, y_points, "-o", color="red")

        x_start = [self._gradient_descent._history_points[0][0]]
        y_start = [self._gradient_descent._history_points[0][1]]
        self.ax.plot(x_start, y_start, "-o", color="pink")

    def add_contours(self, x, y, z):
        self.ax.contour(x, y, z, 20)


def main():
    points: List[Vector2D] = [(1, 0), (-2, -2), (4, 4), (-3, 2), (0, 0), (4, 0)]

    fig, axs = plt.subplots(3, 2, figsize=(8, 10), constrained_layout=True)
    fig.suptitle("Newton method")
    axes = axs.flat

    for i, point in enumerate(points):
        # gradient = SteepestGradientDescent(iterations=200, beta=0.01)
        gradient = NewtonGradientDescent(iterations=100, beta=0.1)
        gradient.run(point)

        print(gradient._history_time_in_microseconds)

        plotter = GradientPlotter(gradient, fig=fig, ax=axes[i])
        plotter.plot()

    plt.show()

if __name__ == "__main__":
    main()
