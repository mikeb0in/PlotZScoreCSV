import csv
import numpy as np
import matplotlib.pyplot as plt
import statistics

def collect_data(file):
    with open(file, newline='') as in_file:
        reader = csv.reader(in_file, delimiter=',', quotechar='"')
        vals = []
        for row in reader:
            print(row)
            # filter shit header rows
            if len(row) == 0 or len(row[0]) != 1 or row[0].isspace():
                continue
            # remove first column that doesn't contain real data
            row.pop(0)

            # convert strings to int
            row = [int(row) for row in row]

            vals += row

        print(vals)
        return vals

def z_score(vals):
    mean = sum(vals) / len(vals)
    std_dev = statistics.stdev(vals)
    print(mean, std_dev)
    z_vals = []
    for val in vals:
        z_val = (val - int(mean)) / int(std_dev)
        z_vals.append(z_val)
    return z_vals

def compute_linear_regression(x_vals, y_vals):
    coef = np.polyfit(x_vals, y_vals, 1)
    print(coef)
    poly1d_fn = np.poly1d(coef)
    # poly1d_fn is now a function which takes in x and returns an estimate for y
    print(poly1d_fn)
    return poly1d_fn

def compute_r_squared(x_vals, y_vals, poly_fn):
    residual_vals_squared = []
    for x_val, y_val in zip(x_vals, y_vals):
        real_value = y_val
        estimated_value = poly_fn(x_val)
        residual_vals_squared.append((real_value - estimated_value) ** 2)
    summed_residual_squared = sum(residual_vals_squared)
    mean_y = sum(y_vals) / len(y_vals)
    actual_vals_squared = []
    for y_val in y_vals:
        actual_vals_squared.append((y_val - mean_y) ** 2)
    summed_actual_suared = sum(actual_vals_squared)
    r_squared = 1 - (summed_residual_squared / summed_actual_suared)
    print(r_squared)
    return r_squared

def plot_data(x_vals, y_vals, poly_fn, r_squared):
    plt.plot(x_vals, y_vals, 'yo', x_vals, poly_fn(x_vals), '--k')  # '--k'=black dashed line, 'yo' = yellow circle marker
    plt.figtext(0.4, 0.7,
                "r^2 = {:.2f}".format(r_squared),
                horizontalalignment="center",
                verticalalignment="center",
                wrap=True, fontsize=14,
                color="green")
    plt.show()


if __name__ == '__main__':
    x_vals = z_score(collect_data('data/x.csv'))
    y_vals = z_score(collect_data('data/y.csv'))
    print(x_vals, y_vals)
    poly_fn = compute_linear_regression(x_vals, y_vals)
    plot_data(x_vals, y_vals, poly_fn, compute_r_squared(x_vals, y_vals, poly_fn))