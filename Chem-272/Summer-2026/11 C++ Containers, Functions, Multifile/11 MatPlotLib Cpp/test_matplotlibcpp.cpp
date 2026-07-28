#include <vector>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main() {
    // Prepare data
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {2.5, 3.8, 1.2, 4.5, 2.0};

    // Create a scatter plot
    plt::scatter(x, y);

    // Add labels and title (optional)
    plt::xlabel("X-axis Label");
    plt::ylabel("Y-axis Label");
    plt::title("Simple Scatter Plot in C++");

    // Show the plot
    plt::show();

    return 0;
}