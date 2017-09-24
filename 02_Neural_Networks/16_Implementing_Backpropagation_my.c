/*                  FINAL VERSION
 * Functions don't allocate arrays that they return. */

/* Matrices are represented as 1-D arrays in memory.
 * That means they are contiguous in memory, flat arrays.
 * Minimum dimension is 1, not 0, and internal dimensions must match. */

/* Assumes a Neural network with only one hidden layer. */

#define BACKPROPAGATION
#ifdef BACKPROPAGATION

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX2(a, b) (((a) > (b)) ? (a) : (b))
#define MAX3(a, b, c) (((MAX2((a), (b))) > (c)) ? (MAX2((a), (b))) : (c))


/* Activation function on a scalar */
inline double sigmoid_scalar(const double x) {
    return 1. / (1. + exp(-x));
}

/* Activation function on an array
   Returns the modified array that it takes as the last argument. */
double *sigmoid(const double *a, const unsigned length, double *result) {
    for (size_t i = 0; i < length; i++) {
        result[i] = 1. / (1. + exp(-a[i]));
    }

    return result;
}

/* Normal (Gaussian) distribution - by the Central Limit Theorem */
/* Lowest quality, plus too slow. An order of magnitude slower
/* than the rest of the implementations. Correct, though. */
void normal_clt(const double mean, const double std, double *arr, const unsigned length) {
    const unsigned n_sum = 25;

    for (size_t i = 0; i < length; i++) {
        double s = 0;
        for (size_t j = 0; j < n_sum; j++) {
            s += (double)rand() / RAND_MAX;
        }
        s -= n_sum / 2.0;
        s /= sqrt(n_sum / 12.0);
        arr[i] = s * std + mean;
    }
}

/* Normal (Gaussian) distribution - Box–Muller transform */
void normal_box_muller(const double mean, const double std, double *arr, const unsigned length) {
    const double two_pi = 8.0 * atan(1.0);
    double x1;
    double x2;

    for (size_t i = 0; i < length; i += 2) {
        double y1;
        double y2;
        x1 = (rand() + 1.) / (RAND_MAX + 2.);
        x2 = rand() / (RAND_MAX + 1.);
        y1 = sqrt(-2.0 * log(x1)) * sin(two_pi * x2);
        y2 = sqrt(-2.0 * log(x1)) * cos(two_pi * x2);
        arr[i] = y1 * std + mean;
        arr[i + 1] = y2 * std + mean;
    }
}

/* Normal (Gaussian) distribution - Marsaglia polar method */
void normal_marsaglia(const double mean, const double std, double *arr, const unsigned length) {
    double x1;
    double x2;

    for (size_t i = 0; i < length; i += 2) {
        double s, y1, y2, f;
        do {
            x1 = 2.0 * rand() / (double)RAND_MAX - 1.0;
            x2 = 2.0 * rand() / (double)RAND_MAX - 1.0;
            s = x1 * x1 + x2 * x2;
        } while (s >= 1.0 || s == 0.0);
        f = sqrt(-2.0 * log(s) / s);
        y1 = x1 * f;
        y2 = x2 * f;
        arr[i] = y1 * std + mean;
        arr[i + 1] = y2 * std + mean;
    }
}

/* Returns the mean value of an array */
double mean(const double *arr, const unsigned length) {
    double sum = 0.;

    for (size_t i = 0; i < length; i++) {
        sum += arr[i];
    }

    return sum / length;
}

/* Dot product of two arrays, a and b, or matrix product
 * Returns an array that's passed in as the last argument, c. */
double *dot(const double *a, const unsigned n_rows_a, const unsigned n_cols_a, \
            const double *b, const unsigned n_rows_b, const unsigned n_cols_b, double *c) {

    /* Check lengths of the input arrays */
    if (n_cols_a != n_rows_b) {
        printf("#columns A must be equal to #rows B!\n");
        system("pause");
        exit(-2);
    }

    for (size_t i = 0; i < n_rows_a; i++) {
        for (size_t k = 0; k < n_cols_b; k++) {
            double sum = 0.0;
            for (size_t j = 0; j < n_cols_a; j++) {
                sum += a[i*n_cols_a + j] * b[j*n_cols_b + k];
            }
            c[i*n_cols_b + k] = sum;
        }
    }

    return c;
}

/*  Adds two arrays, element-wise, and puts the result in an array
    that is passed in as the last argument, and also returns it.
    Arrays must be of the same length, or, one of them, or both, can be scalars.
    Use 0 as the length of a scalar, and pass its address in (a pointer to it). */
double *add_arrays(const double *a, const unsigned n_a, const double *b, const unsigned n_b, double *result) {
    /* Check lengths of the input arrays */
    if ((n_a != n_b) && (n_a != 0) && (n_b != 0)) {
        printf("Length of A must be equal to length of B!\n");
        system("pause");
        exit(-2);
    }

    /* Size of result is maximum of n_a and n_b. */
    unsigned size = n_a > n_b ? n_a : n_b;

    /* Both a and b are scalars. */
    if (size == 0) {
        result[0] = *a + *b;
    }
    /* Only a is scalar. */
    else if (n_a == 0) {
        for (size_t i = 0; i < n_b; i++) {
            result[i] = *a + b[i];
        }
    }
    /* Only b is scalar. */
    else if (n_b == 0) {
        for (size_t i = 0; i < n_a; i++) {
            result[i] = a[i] + *b;
        }
    }
    /* Neither a nor b are scalars. */
    else {
        for (size_t i = 0; i < n_a; i++) {
            result[i] = a[i] + b[i];
        }
    }

    return result;
}

/*  Subtracts the second array from the first one, element-wise, and puts the result
    in an array that is passed in as the last argument, and also returns it.
    Arrays must be of the same length, or, one of them, or both, can be scalars.
    Use 0 as the length of a scalar, and pass its address in (a pointer to it). */
double *subtract_arrays(const double *a, const unsigned n_a, const double *b, const unsigned n_b, double *result) {
    /* Check lengths of the input arrays */
    if ((n_a != n_b) && (n_a != 0) && (n_b != 0)) {
        printf("Length of A must be equal to length of B!\n");
        system("pause");
        exit(-2);
    }

    /* Size of result is maximum of n_a and n_b. */
    unsigned size = n_a > n_b ? n_a : n_b;

    /* Both a and b are scalars. */
    if (size == 0) {
        result[0] = *a - *b;

    }
    /* Only a is scalar. */
    else if (n_a == 0) {
        for (size_t i = 0; i < n_b; i++) {
            result[i] = *a - b[i];
        }
    }
    /* Only b is scalar. */
    else if (n_b == 0) {
        for (size_t i = 0; i < n_a; i++) {
            result[i] = a[i] - *b;
        }
    }
    /* Neither a nor b are scalars. */
    else {
        for (size_t i = 0; i < n_a; i++) {
            result[i] = a[i] - b[i];
        }
    }

    return result;
}

/*  Multiplies two arrays, element-wise, and puts the result in an array
    that is passed in as the last argument, and also returns it.
    Arrays must be of the same length, or, one of them, or both, can be scalars.
    Use 0 as the length of a scalar, and pass its address in (a pointer to it). */
double *multiply_arrays(const double *a, const unsigned n_a, const double *b, const unsigned n_b, double *result) {
    /* Check lengths of the input arrays */
    if ((n_a != n_b) && (n_a != 0) && (n_b != 0)) {
        printf("Length of A must be equal to length of B!\n");
        system("pause");
        exit(-2);
    }

    /* Size of result is maximum of n_a and n_b. */
    unsigned size = n_a > n_b ? n_a : n_b;

    /* Both a and b are scalars. */
    if (size == 0) {
        result[0] = *a * *b;
    }
    /* Only a is scalar. */
    else if (n_a == 0) {
        for (size_t i = 0; i < n_b; i++) {
            result[i] = *a * b[i];
        }
    }
    /* Only b is scalar. */
    else if (n_b == 0) {
        for (size_t i = 0; i < n_a; i++) {
            result[i] = a[i] * *b;
        }
    }
    /* Neither a nor b are scalars. */
    else {
        for (size_t i = 0; i < n_a; i++) {
            result[i] = a[i] * b[i];
        }
    }

    return result;
}

/*  Multiplies two arrays, element-wise, and puts the result in an array
    that is passed in as the last argument, and also returns it.
    Arrays must be of the same length, or, one of them, or both, can be scalars.
    Use 0 as the length of a scalar, and pass its address in (a pointer to it). */
double *divide_arrays(const double *a, const unsigned n_a, const double *b, const unsigned n_b, double *result) {
    /* Check lengths of the input arrays */
    if ((n_a != n_b) && (n_a != 0) && (n_b != 0)) {
        printf("Length of A must be equal to length of B!\n");
        system("pause");
        exit(-2);
    }

    /* Size of result is maximum of n_a and n_b. */
    unsigned size = n_a > n_b ? n_a : n_b;

    /* Both a and b are scalars. */
    if (size == 0) {
        result[0] = *a / *b;
    }
    /* Only a is scalar. */
    else if (n_a == 0) {
        for (size_t i = 0; i < n_b; i++) {
            result[i] = *a / b[i];
        }
    }
    /* Only b is scalar. */
    else if (n_b == 0) {
        for (size_t i = 0; i < n_a; i++) {
            result[i] = a[i] / *b;
        }
    }
    /* Neither a nor b are scalars. */
    else {
        for (size_t i = 0; i < n_a; i++) {
            result[i] = a[i] / b[i];
        }
    }

    return result;
}

/*  Updates an array, element-wise, by adding another array to it.
    Takes both arrays in, and returns the updated one (the first one).
    The return value (address of the first array) doesn't have to be used.
    Arrays must be of the same length, or, the second one can be a scalar.
    Use 0 as the length of a scalar, and pass its address in (a pointer to it). */
double *add_update(double *a, const unsigned n_a, const double *b, const unsigned n_b) {
    /* Check lengths of the input arrays */
    if (n_a == 0) {
        printf("'A' cannot be a scalar!\n");
        system("pause");
        exit(-2);
    }
    if ((n_a != n_b) && (n_b != 0)) {
        printf("Length of A must be equal to length of B!\n");
        system("pause");
        exit(-2);
    }

    /* b is scalar */
    if (n_b == 0) {
        for (size_t i = 0; i < n_a; i++) {
            a[i] += *b;
        }
    }
    /* b is array */
    else {
        for (size_t i = 0; i < n_a; i++) {
            a[i] += b[i];
        }
    }

    return a;
}

/*  Compares two arrays element-wise, and puts the result in an array
    that is passed in as the last argument, and also returns it.
    If an element of array a is greater than a corresponding element of
    array b, the resulting array will have 1.0 in that position;
    it will have 0.0 otherwise.
    Arrays must be of the same length, or, one of them, or both, can be scalars.
    Use 0 as the length of a scalar, and pass its address in (a pointer to it). */
double *greater_than(const double *a, const unsigned n_a, const double *b, const unsigned n_b, double *result) {
    /* Check lengths of the input arrays */
    if ((n_a != n_b) && (n_a != 0) && (n_b != 0)) {
        printf("Length of A must be equal to length of B!\n");
        system("pause");
        exit(-2);
    }

    /* Size of result is maximum of n_a and n_b. */
    unsigned size = n_a > n_b ? n_a : n_b;

    /* Both a and b are scalars. */
    if (size == 0) {
        result[0] = *a > *b;
    }
    /* Only a is scalar. */
    else if (n_a == 0) {
        for (size_t i = 0; i < n_b; i++) {
            result[i] = *a > b[i];
        }
    }
    /* Only b is scalar. */
    else if (n_b == 0) {
        for (size_t i = 0; i < n_a; i++) {
            result[i] = a[i] > *b;
        }
    }
    /* Neither a nor b are scalars. */
    else {
        for (size_t i = 0; i < n_a; i++) {
            result[i] = a[i] > b[i];
        }
    }

    return result;
}

/*  Compares two arrays element-wise, and puts the result in an array
    that is passed in as the last argument, and also returns it.
    If an element of array a is equal to a corresponding element of
    array b, the resulting array will have 1.0 in that position;
    it will have 0.0 otherwise.
    Arrays must be of the same length, or, one of them, or both, can be scalars.
    Use 0 as the length of a scalar, and pass its address in (a pointer to it). */
double *equal(const double *a, const unsigned n_a, const double *b, const unsigned n_b, double *result) {
    /* Check lengths of the input arrays */
    if ((n_a != n_b) && (n_a != 0) && (n_b != 0)) {
        printf("Length of A must be equal to length of B!\n");
        system("pause");
        exit(-2);
    }

    /* Size of result is maximum of n_a and n_b. */
    unsigned size = n_a > n_b ? n_a : n_b;

    /* Both a and b are scalars. */
    if (size == 0) {
        result[0] = *a == *b;
    }
    /* Only a is scalar. */
    else if (n_a == 0) {
        for (size_t i = 0; i < n_b; i++) {
            result[i] = *a == b[i];
        }
    }
    /* Only b is scalar. */
    else if (n_b == 0) {
        for (size_t i = 0; i < n_a; i++) {
            result[i] = a[i] == *b;
        }
    }
    /* Neither a nor b are scalars. */
    else {
        for (size_t i = 0; i < n_a; i++) {
            result[i] = a[i] == b[i];
        }
    }

    return result;
}

/*  Takes and returns a new matrix, t, which is a transpose of the original one, m.
    It's also flat in memory, i.e., 1-D, but it should be looked at as a transpose
    of m, meaning, n_rows_t == n_cols_m, and n_cols_t == n_rows_m.
    The original matrix m stays intact. */
double *transpose(const double *m, const unsigned n_rows_m, const unsigned n_cols_m, double *t) {
    for (size_t i = 0; i < n_rows_m; i++) {
        for (size_t j = 0; j < n_cols_m; j++) {
            t[j*n_rows_m + i] = m[i*n_cols_m + j];
        }
    }

    /* Visual validation - Prints t like m, the original */
    const int validate = 0;
    if (validate) {
        for (size_t i = 0; i < n_rows_m; i++) {
            for (size_t j = 0; j < n_cols_m; j++) {
                printf("%8.3f ", t[j*n_rows_m + i]);
            }
            printf("\n");
        }
        printf("\n");
    }

    return t;
}

/* Prints vector, or matrix. */
void print(const double *m, const unsigned n_rows_m, const unsigned n_cols_m) {
    for (size_t i = 0; i < n_rows_m; i++) {
        for (size_t j = 0; j < n_cols_m; j++) {
            printf("%8.3f ", m[i*n_cols_m + j]);
        }
        printf("\n");
    }
    printf("\n");
}

/* Reads inputs to NN from a text file, for training it or testing it,
   and returns them as an array of doubles of n_records * n_features size.
   Creates the array.
   n_records is the number of the training or testing pairs (input, correct output).
   n_features is the number of inputs to the NN. */
double *read_features(const char *file_name, const unsigned n_records, const unsigned n_features) {
    double *features = malloc(n_records * n_features * sizeof(*features));
    if (features == NULL) {
        printf("Couldn't allocate memory!\n");
        system("pause");
        exit(-1);
    }

    FILE *file = fopen(file_name, "r");
    if (file == NULL) {
        printf("Couldn't open file \"%s\"", file_name);
        perror(" ");
        system("pause");
        exit(-3);
    }

    for (size_t i = 0; i < n_records * n_features; i++) {
        fscanf(file, "%lf", &features[i]);
    }

    /* Visual validation */
    const int validate = 0;
    if (validate) {
        for (size_t i = 0; i < n_records * n_features; ) {
            printf("%21.17f", features[i]);
            if (++i % n_features == 0) {
                printf("\n");
            }
        }
        printf("\n");
    }

    fclose(file);

    return features;
}

/* Reads targets for NN from a text file, for training it or testing it,
   and returns them as an array of doubles of n_records size.
   Creates the array.
   n_records is the number of the training or testing pairs (input, correct output).
   n_outputs is the number of outputs from the NN. */
double *read_targets(const char *file_name, const unsigned n_records, const unsigned n_outputs) {
    double *targets = malloc(n_records * n_outputs * sizeof(*targets));
    if (targets == NULL) {
        printf("Couldn't allocate memory!\n");
        system("pause");
        exit(-1);
    }

    FILE *file = fopen(file_name, "r");
    if (file == NULL) {
        printf("Couldn't open file \"%s\"", file_name);
        perror(" ");
        system("pause");
        exit(-3);
    }

    for (size_t i = 0; i < n_records * n_outputs; i++) {
        fscanf(file, "%lf", &targets[i]);
    }

    /* Visual validation */
    const int validate = 0;
    if (validate) {
        for (size_t i = 0; i < n_records * n_outputs; ) {
            printf("%1.0f\n", targets[i++]);
        }
        printf("\n");
    }

    fclose(file);

    return targets;
}

/* Trains the NN.
 * n_records is the number of the training pairs (input, correct output).
 * n_features is the number of inputs to the NN, and it's the same thing as w_i_h_rows.
 * w_i_h_cols is the number of units in the hidden layer, and it's the same as w_h_o_rows.
 * w_h_o_cols is the number of output units, in the only output layer. */
void train_nn(const unsigned n_epochs, const double learn_rate, \
    const unsigned n_records, const unsigned n_features, \
    double *weights_input_hidden, const unsigned w_i_h_rows, const unsigned w_i_h_cols,\
    double *weights_hidden_output, const unsigned w_h_o_rows, const unsigned w_h_o_cols) {

    /* For measuring time */
    clock_t t0, t1;
    float diff;

    const double one = 1.0;
    const double learn_rate_over_n_records = learn_rate / (double)n_records;
    double last_loss = 0.0;
    double *features = NULL;                                                                        // shape (360, 6)
    double *targets = NULL;                                                                         // shape (360,)

    features = read_features("features.txt", n_records, n_features);
    targets = read_targets("targets.txt", n_records, w_h_o_cols);

    t0 = clock();

    /* This would be faster on stack, but it wouldn't fit on stack in case of
     * large NNs and/or large data sets. Of course, this code can be sped up
     * by calculating sizes only once, and storing them in variables. */

    double *delta_w_i_h = malloc(w_i_h_rows * w_i_h_cols * sizeof(*delta_w_i_h));                   // shape (6, 2)
    double *delta_w_h_o = malloc(w_h_o_rows * w_h_o_cols * sizeof(*delta_w_h_o));                   // shape (2,)
    double *hidden_input = malloc(n_records * w_i_h_cols * sizeof(*hidden_input));                  // shape (360, 2)
    double *hidden_output = malloc(n_records * w_i_h_cols * sizeof(*hidden_output));                // shape (360, 2)
    double *output_input = malloc(n_records * w_h_o_cols * sizeof(*output_input));                  // shape (360,)
    double *output_output = malloc(n_records * w_h_o_cols * sizeof(*output_output));                // shape (360,)
    double *error = malloc(n_records * w_h_o_cols * sizeof(*error));                                // shape (360,)
    double *output_error_term = malloc(n_records * w_h_o_cols * sizeof(*output_error_term));        // shape (360,)
    double *hidden_error_term = malloc(n_records * w_i_h_cols * sizeof(*hidden_error_term));        // shape (360, 2)
    double *tmp1 = malloc(n_records * MAX3(n_features, w_i_h_cols, w_h_o_cols) * sizeof(*tmp1));    // shape (360, 6)
    double *tmp2 = malloc(n_records * MAX3(n_features, w_i_h_cols, w_h_o_cols) * sizeof(*tmp2));    // shape (360, 6)
    
    for (size_t epoch = 0; epoch < n_epochs; epoch++) {
        double mse = 0;                                                                                                         // shape (), that is, a scalar

        /*** Forward pass ***/
        
        hidden_input = dot(features, n_records, n_features, weights_input_hidden, w_i_h_rows, w_i_h_cols, hidden_input);        // shape (360, 2)
        hidden_output = sigmoid(hidden_input, n_records * w_i_h_cols, hidden_output);                                           // shape (360, 2)

        output_input = dot(hidden_output, n_records, w_i_h_cols, weights_hidden_output, w_h_o_rows, w_h_o_cols, output_input);  // shape (360,)
        output_output = sigmoid(output_input, n_records * w_h_o_cols, output_output);                                           // shape (360,)
        
        /*** Backward pass ***/

        error = subtract_arrays(targets, n_records * w_h_o_cols, output_output, n_records * w_h_o_cols, error);                 // shape (360,)

        tmp1 = subtract_arrays(&one, 0, output_output, n_records * w_h_o_cols, tmp1);                                           // shape (360,)
        tmp2 = multiply_arrays(output_output, n_records * w_h_o_cols, tmp1, n_records * w_h_o_cols, tmp2);                      // shape (360,)
        output_error_term = multiply_arrays(error, n_records * w_h_o_cols, tmp2, n_records * w_h_o_cols, output_error_term);    // shape (360,)

        /* Propagate errors to hidden layer */
        tmp1 = subtract_arrays(&one, 0, hidden_output, n_records * w_i_h_cols, tmp1);                                           // shape (360, 2)
        tmp2 = multiply_arrays(hidden_output, n_records * w_i_h_cols, tmp1, n_records * w_i_h_cols, tmp2);                      // shape (360, 2)
        tmp1 = dot(output_error_term, n_records, w_h_o_cols, weights_hidden_output, w_h_o_cols, w_h_o_rows, tmp1);              // shape (360, 2)
        /* Calculate the error term for the hidden layer */
        hidden_error_term = multiply_arrays(tmp1, n_records * w_h_o_rows, tmp2, n_records * w_i_h_cols, hidden_error_term);     // shape (360, 2)

        /* Update the change in weights */
        tmp1 = transpose(features, n_records, n_features, tmp1);                                                                // shape (6, 360)
        delta_w_i_h = dot(tmp1, n_features, n_records, hidden_error_term, n_records, w_h_o_rows, delta_w_i_h);                  // shape (6, 2) - The gradient descent step, the error times the gradient times the inputs
        delta_w_h_o = dot(output_error_term, w_h_o_cols, n_records, hidden_output, n_records, w_i_h_cols, delta_w_h_o);         // shape (2,) - The gradient descent step, the error times the gradient times the inputs
        
        /* Update weights */
        tmp1 = multiply_arrays(&learn_rate_over_n_records, 0, delta_w_i_h, n_features * w_h_o_rows, tmp1);                      // shape (6, 2)
        weights_input_hidden = add_update(weights_input_hidden, w_i_h_rows * w_i_h_cols, tmp1, n_features * w_h_o_rows);        // shape (6, 2)
        tmp1 = multiply_arrays(&learn_rate_over_n_records, 0, delta_w_h_o, w_h_o_cols * w_i_h_cols, tmp1);                      // shape (2,)
        weights_hidden_output = add_update(weights_hidden_output, w_h_o_rows * w_h_o_cols, tmp1, w_h_o_cols * w_i_h_cols);      // shape (2,)
        
        tmp2 = multiply_arrays(error, n_records * w_h_o_cols, error, n_records * w_h_o_cols, tmp2);                             // shape (360,)
        mse = .5 * mean(tmp2, n_records * w_h_o_cols);                                                                          // shape (), that is, a scalar

        /* Printing out the mean square error on the training set */
        if (epoch % (n_epochs / 10) == 0) {
            const double loss = mse;
            if (last_loss && (last_loss < loss)) {
                printf("Train loss: %f  WARNING - Loss Increasing\n", loss);
            }
            else {
                printf("Train loss: %f\n", loss);
            }
            last_loss = loss;
        }

        //print(delta_w_i_h, w_i_h_rows, w_i_h_cols);
        //print(delta_w_h_o, w_h_o_rows, w_h_o_cols);
        //break;
    }

    free(delta_w_i_h);
    free(delta_w_h_o);
    free(hidden_input);
    free(hidden_output);
    free(output_input);
    free(output_output);
    free(error);
    free(output_error_term);
    free(hidden_error_term);
    free(tmp1);
    free(tmp2);

    t1 = clock();
    diff = (float)(t1 - t0) / CLOCKS_PER_SEC;
    printf("Training took %.3lf s\n", diff);

    free(features);
    free(targets);
}

/* Calculates accuracy on test data.
 * n_records is the number of the testing pairs (input, correct output).
 * n_features is the number of inputs to the NN, and it's the same thing as w_i_h_rows.
 * w_i_h_cols is the number of units in the hidden layer, and it's the same as w_h_o_rows.
 * w_h_o_cols is the number of output units, in the only output layer. */
double test_nn(const unsigned n_records, const unsigned n_features, \
    const double *weights_input_hidden, const unsigned w_i_h_rows, const unsigned w_i_h_cols, \
    const double *weights_hidden_output, const unsigned w_h_o_rows, const unsigned w_h_o_cols) {

    /* For measuring time */
    clock_t t0, t1;
    float diff;

    const double one_half = 0.5;
    double accuracy = 0.0;
    double *features_test = NULL;                                                       // shape (40, 2)
    double *targets_test = NULL;                                                        // shape (40,)

    features_test = read_features("features_test.txt", n_records, n_features);
    targets_test = read_targets("targets_test.txt", n_records, w_h_o_cols);

    t0 = clock();

    double *hidden_input = malloc(n_records * w_i_h_cols * sizeof(*hidden_input));      // shape (40, 2)
    double *hidden_output = malloc(n_records * w_i_h_cols * sizeof(*hidden_output));    // shape (40, 2)
    double *output_input = malloc(n_records * w_h_o_cols * sizeof(*output_input));      // shape (40,)
    double *output_output = malloc(n_records * w_h_o_cols * sizeof(*output_output));    // shape (40,)
    double *predictions = malloc(n_records * w_h_o_cols * sizeof(*output_output));      // shape (40,)
    double *tmp = malloc(n_records * w_h_o_cols * sizeof(*output_output));              // shape (40,)

    hidden_input = dot(features_test, n_records, n_features, weights_input_hidden, w_i_h_rows, w_i_h_cols, hidden_input);
    hidden_output = sigmoid(hidden_input, n_records * w_i_h_cols, hidden_output);

    output_input = dot(hidden_output, n_records, w_i_h_cols, weights_hidden_output, w_h_o_rows, w_h_o_cols, output_input);
    output_output = sigmoid(output_input, n_records * w_h_o_cols, output_output);
    
    predictions = greater_than(output_output, n_records * w_h_o_cols, &one_half, 0, predictions);
    
    tmp = equal(predictions, n_records * w_h_o_cols, targets_test, n_records * w_h_o_cols, tmp);
    accuracy = mean(tmp, n_records * w_h_o_cols);

    free(hidden_input);
    free(hidden_output);
    free(output_input);
    free(output_output);
    free(predictions);
    free(tmp);

    t1 = clock();
    diff = (float)(t1 - t0) / CLOCKS_PER_SEC;
    printf("Testing took %.3lf s\n", diff);

    free(features_test);
    free(targets_test);

    return accuracy;
}

int main(int argc, char *argv[]) {
    /* Intializes random number generator */
    time_t t;
    srand((unsigned)time(&t));
    srand(21);

    /* Neural Network hyperparameters */
    const unsigned n_hidden = 2;        // Number of hidden units. There is only one hidden layer.
    const unsigned n_output = 1;        // Number of output units, in the only output layer.
    const unsigned n_epochs = 900;
    const double learn_rate = .005;     // Eta

    /* Features are inputs to our NN, and there's 6 of them.
     * We have 360 train data points (records).
     * We also have 40 test points (records). */
    const unsigned n_features = 6;
    const unsigned n_train = 360;    
    const unsigned n_test = 40;

    /* Function pointer to Normal distribution functions */
    void(*fp)(const double, const double, double*, const unsigned) = normal_marsaglia;

    /* Initial weights - They should be small and random, around 0, so that inputs are in the linear region of the sigmoid. */
    double *weights_input_hidden = malloc(n_features * n_hidden * sizeof(*weights_input_hidden));       // shape (6, 2)
    double *weights_hidden_output = malloc(n_hidden * n_output * sizeof(*weights_hidden_output));       // shape (2,)
    if (!weights_input_hidden || !weights_hidden_output) {
        printf("Couldn't allocate memory!\n");
        exit(-1);
    }
    fp(0.0, 1. / sqrt(n_features), weights_input_hidden, n_features*n_hidden);
    fp(0.0, 1. / sqrt(n_features), weights_hidden_output, n_hidden);

    train_nn(n_epochs, learn_rate, n_train, n_features, weights_input_hidden, n_features, n_hidden, weights_hidden_output, n_hidden, n_output);
    double accuracy = test_nn(n_test, n_features, weights_input_hidden, n_features, n_hidden, weights_hidden_output, n_hidden, n_output);

    printf("Prediction accuracy: %.3f\n", accuracy);

    printf("\n");
    printf("weights_input_hidden: \n");
    print(weights_input_hidden, n_features, n_hidden);
    printf("weights_hidden_output: \n");
    print(weights_hidden_output, n_hidden, n_output);

    free(weights_hidden_output);
    free(weights_input_hidden);

    system("pause");
    return(0);
}

#endif  // BACKPROPAGATION

// Neural Network hyperparameters 5-6, 1, 1000-2000 and 1.005 give accuracy of 0.750, even with normal_clt.
// Default values of 2, 1, 900, .005 give accuracy of 0.650.
