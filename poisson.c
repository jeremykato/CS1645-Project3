#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>

#define POINT(x, y) (x + (y * (n + 2)))


void print_matrix(double* matrix, int n, int m) {
    for (int j = (m - 1); j >= 0; j--) {
        for (int i = (n - 1); i >= 0; i--) {
            printf("%2.6f\t", matrix[POINT(i, j)]);
        }
        printf("\n");
    }
    printf("\n");
}

double get_max_error(double* appx, double* exact, int n, int m) {
    
    double max_error = 0.0, curr_error = 0.0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            curr_error = fabs(appx[POINT(i, j)] - exact[POINT(i, j)]);
            if (curr_error > max_error) {
                max_error = curr_error;
            }
        }
    }

    return max_error;

}


int jacobi_iteration(double x_min, double y_min, double x_max, double y_max,
    int n, int m, double delta_change, double *t_appx) {

    double max_change = 1.0; // once this is less than the delta_change, we finish
    int iterations = 0; // number of iterations we take before we're finished

    // the way our jacobi iteration runs, we need a separate data structure to put our
    // updated values into
    double *t_appx_new = calloc(n * m, sizeof(double));
    memcpy(t_appx_new, t_appx, n * m * sizeof(double));
    //double x, y; // coordinates

    // loop while the max change is fewer than our specified delta
    while (max_change > delta_change) {

        max_change = 0.0;
        // begin looping, starting at y_min + (j / y_size)
        // j starts at 1 and ends at m - 1 since j = 0 and j = m - 1 are the boundaries
        for (int j = 1; j < (m - 1); j++) {

            for (int i  = 1; i < (n - 1); i++) {

                //printf("\tAt %d, %d (%d = %d + %d + %d + %d)\n", i, j, POINT(i, j), POINT( (i - 1), j), POINT((i + 1), j), POINT(i, (j - 1)), POINT(i, (j + 1)));
                //printf("\t%2.3f + %2.3f + %2.3f + %2.3f\n", t_appx[POINT((i - 1), j)], t_appx[POINT((i + 1), j)], t_appx[POINT(i, (j - 1))], t_appx[POINT(i, (j + 1))]);

                t_appx_new[POINT(i, j)] = 0.25 * (t_appx[POINT((i - 1), j)] + t_appx[POINT((i + 1), j)] + t_appx[POINT(i, (j - 1))] + t_appx[POINT(i, (j + 1))]);

                double change = fabs(t_appx_new[POINT(i, j)] - t_appx[POINT(i, j)]);

                //printf("(%d, %d): %2.7f\n", i, j, t_appx_new[POINT(i, j)]);


                if (change > max_change) {
                    max_change = change;
                }

            }
            //printf("\n");

        }
        iterations++;

        // copy old into new
        memcpy(t_appx, t_appx_new, n * m * sizeof(double));
        //print_matrix(t_appx, n, m);

    }

    // if we reach here, we're done!
    return iterations;

}

int main(int argc, char *argv[]) {

    if (argc != 10) {
        printf("Error: 9 arguments are necessary.\n");
        printf("program-name <X-Min> <X-Max> <Y-Min> <Y-Max> <Grid Width Per Thread> <Grid Height Per Thread>\n"\
        "      <Convergence-Value> <Threads in X Direction> <Threads in Y Direction>\n");
        exit(-1);
    }

    const double    x_min = atof(argv[1]), x_max = atof(argv[2]), 
                    y_min = atof(argv[3]), y_max = atof(argv[4]);

    const int       n = atoi(argv[5]), m = atoi(argv[6]); // per thread

    const double    delta_change = atof(argv[7]);

    const double    x_size = fabs(x_max - x_min),
                    y_size = fabs(y_max - y_min);

    const int       threads_x = atoi(argv[8]), threads_y = atoi(argv[9]),
                    total_threads = threads_x * threads_y;

    const int       total_n = (threads_x * n) + 2, total_m = (threads_y * m) + 2,
                    total_grid_size = total_n * total_m;

    // temperature grid for the approximated solution
    // and the exact solution
    double *t_appx  = (double*) calloc(total_grid_size, sizeof(double));
    double *t_exact = (double*) calloc(total_grid_size, sizeof(double));

    // boundary conditions for the domain of interest
    double x, y;
    for (int i = 0; i < n + 2; i++) {
        x = (((double) i / (double) (n - 1)) * x_size) + x_min; // x
        t_appx[POINT(i, 0)] = x;                                // T(x, 0) = x
        t_appx[POINT(i, (m - 1))] = x * exp(1);                 // T(x, 1) = x * e
    }

    for (int i = 0; i < m + 2; i++) {
        y = (((double) i / (double) (m - 1)) * y_size) + y_min; // y
        t_appx[POINT(0, i)] = 0;                                // T(0, y) = 0
        t_appx[POINT((n - 1), i)] = 2 * exp(y);                 // T(2, y) = 2 * e^y
    }

    // pthreads go here

    int iterations = jacobi_iteration(x_min, y_min, x_max, y_max, n, m, delta_change, t_appx);

    // compute the exact values along the domain
    // as a note, it would be faster to simply calculate each x and y value, store them,
    // and then multiply them together to avoid having to recalculate each value between
    // x/y min and max, but this will do for now (and for the exact answer, we worry about 
    // performance slightly less)
    for (int i = 0; i < n; i++) {

        x = (((double) i / (double) (n - 1)) * x_size) + x_min; // x

        for (int j = 0; j < m; j++) {

            y = (((double) j / (double) (m - 1)) * y_size) + y_min; // y
            t_exact[POINT(i, j)] = x * exp(y);

        }

    }

    printf("Iterations taken:\t%d\n", iterations);
    printf("Max error: \t\t%2.6f\n", get_max_error(t_appx, t_exact, n, m));

    // printf("Point\t\tExact\t\tAppx.\n");
    // // print out the results
    // for (int j = 0; j < m; j++) {

    //     y = (((double) j / (double) (m - 1)) * y_size) + y_min; // y

    //     for (int i = 0; i < n; i++) {

    //         x = (((double) i / (double) (n - 1)) * x_size) + x_min; // x


    //         printf("(%1.2f, %1.2f)\t%2.6f\t\t%2.6f\n", x, y, 
    //             t_exact[POINT(i, j)], t_appx[POINT(i, j)]);
            
    //     }

    // }

    //print_matrix(t_appx, n, m);
    //print_matrix(t_exact, n, m);
}