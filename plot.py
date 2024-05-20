import cv2
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def get_curve_points(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Image not found or unable to read")

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (1,9),0)

    # Invert the image (if necessary) so the curve is white on black
    _, inverted_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY_INV)

    # Detect edges using Canny (or any other method to detect the curve)
    edges = cv2.Canny(inverted_image, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        raise ValueError("No contours found in the image")

    # Assuming the largest contour is the curve
    curve_contour = max(contours, key=cv2.contourArea)

    # Extract x and y coordinates of the contour points
    points = curve_contour.squeeze()  # Removing unnecessary dimensions

    # Treat the bottom-left corner as origin (flip y-axis and scale)
    height = image.shape[0]
    scale = 1  # 10 pixels = 1 unit

    x_coords = points[:, 0] / scale
    y_coords = (height - points[:, 1]) / scale

    return x_coords, y_coords



def interpolate_points(x_coords, y_coords, num_points=1000):
    # Calculate cumulative distance along the curve
    distance = np.cumsum(np.sqrt(np.diff(x_coords) ** 2 + np.diff(y_coords) ** 2))
    distance = np.insert(distance, 0, 0) / distance[-1]

    # Create linear interpolation function
    interpolator = interp1d(distance, np.vstack((x_coords, y_coords)),kind="linear", axis=1)

    # Generate evenly spaced points along the curve
    alpha = np.linspace(0, 1, num_points)
    interpolated_points = interpolator(alpha)

    return interpolated_points[0], interpolated_points[1]


def plot_curve(x_coords, y_coords):
    # Create a plot
    plt.figure(figsize=(8, 6))

    # Plot the curve with a reduced marker size and dotted line style
    plt.plot(x_coords, y_coords, linestyle='-', color='r', markersize=1)

    # Set the axis labels
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')

    # Set the title of the plot
    plt.title('Curve Plot')

    # Show the grid
    # plt.grid(True)

    # Display the plot
    plt.show()


def main():
    x_cordi = []
    y_cordi = []
    points=[]
    no_of_points = 0
    image_path = 'img_4.png'  # Replace with your image path
    x_coords, y_coords = get_curve_points(image_path)

    # Interpolate points to reduce gaps
    x_coords, y_coords = interpolate_points(x_coords, y_coords, num_points=5000)

    # Print the coordinates
    for x, y in zip(x_coords, y_coords):
        x_cordi.append(round(x,2))
        y_cordi.append(round(y,2))
        points.append((round(x,2),round(y,2)))
        no_of_points += 1

    return (x_cordi, y_cordi, no_of_points,points)


if __name__ == "__main__":
    (x_pt, y_pts, count,points) = main()
    final=[]

    print(x_pt, "\n")
    print(y_pts, "\n")
    print("The total number of points are:", count)
    print("\n",points)
    plot_curve(x_pt, y_pts)
