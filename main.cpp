#include <iostream>
#include <string>
#include <algorithm>
#include <vector>

#include "ortools/lp_data/mps_reader.h"
#include "ortools/base/stringpiece_utils.h"
#include "ortools/glop/lp_solver.h"
#include "ortools/glop/parameters.pb.h"
#include "ortools/lp_data/lp_types.h"

#include "armadillo"

using namespace std;
using namespace arma;

using operations_research::glop::LinearProgram;
using operations_research::glop::MPSReader;
using operations_research::glop::ProblemStatus;
using operations_research::glop::ToDouble;
using operations_research::glop::LPSolver;
using operations_research::glop::GetProblemStatusString;

using operations_research::glop::DenseRow;
using operations_research::glop::SparseMatrix;
using operations_research::glop::RowIndex;
using operations_research::glop::ColIndex;
using operations_research::glop::Fractional;
using operations_research::glop::kInfinity;

int MPSToMatrix(LinearProgram& linear_program, string file_name,
        mat& A, colvec& b, rowvec& c) {
    // Load MPS file
    MPSReader mps_reader;

    if (strings::EndsWith(file_name, ".mps")) {
        if (!mps_reader.LoadFileAndTryFreeFormOnFail(file_name,
                                                   &linear_program)) {
        cerr << "Failed to read problem from file!";
        return EXIT_FAILURE;
      }
    } else {
        cerr << "Input file extension must be .mps!";
        return EXIT_FAILURE;
    }

    // Extract information
    
    // c <- objective function coefficients
    DenseRow coeffs = linear_program.objective_coefficients();
    // rowvec c(coeffs.size().value());
    c = resize(c, 1, coeffs.size().value());
    int i = 0;
    for (double coeff: coeffs) {
        c[i] = coeff;
        i++;
    }

    // A <- coefficients of constraints
    const SparseMatrix& matrix = linear_program.GetSparseMatrix();
    // mat A(matrix.num_rows().value(), matrix.num_cols().value());
    A = resize(A, matrix.num_rows().value(), matrix.num_cols().value());
    for (RowIndex row(0); row < matrix.num_rows(); ++row) {
        for (ColIndex col(0); col < matrix.num_cols(); ++col) {
            A(row.value(), col.value()) = ToDouble(matrix.LookUpValue(row, col));
        }
    }

    // b <- bounds
    const RowIndex num_rows = linear_program.num_constraints();
    b = resize(b, num_rows.value(), 1);
    for (RowIndex row(0); row < num_rows; ++row) {
        const Fractional lower_bound = linear_program.constraint_lower_bounds()[row];
        const Fractional upper_bound = linear_program.constraint_upper_bounds()[row];
        
        // Transformations
        if (lower_bound == -kInfinity) {
            A.insert_cols(A.n_cols, 1);
            A(row.value(), A.n_cols - 1) = 1;

            c.insert_cols(c.n_cols, 1);

            b[row.value()] = ToDouble(upper_bound);
        } else if (upper_bound == kInfinity) {
            A.row(row.value()).transform( [&](double val) { return (-val); } );
            A.insert_cols(A.n_cols, 1);
            A(row.value(), A.n_cols - 1) = 1;

            c.insert_cols(c.n_cols, 1);

            b[row.value()] = -ToDouble(lower_bound);
        } else {
            b[row.value()] = ToDouble(upper_bound);
        }
    }

    return 0;
}

void print_solution(mat A, mat d, vector<int> B_index, double z) {
	cout << endl;
	cout << "*********" << endl;
	cout << "Solved!" << endl;
	cout << "*********" << endl;
	cout << endl;

	A.print("A:");

	d.print("dT:");

	cout << "Variables: " << endl;
	int i = 0;
	for (int ind: B_index) {
		cout << ind << ": " << A(i, A.n_cols - 1) << endl;
		++i;
	}
	cout << endl;

	cout << "Z = " << z << endl;
}

void find_pivot(mat A, vector<int> B_index, int q, int& row, double& t) {
	mat alpha = A.col(q);
	mat beta  = A.col(A.n_cols - 1);

	alpha.print("alpha:");
	beta.print("beta:");
	A.print("A:");

	double min = datum::inf;
	int ind = -1;
	for (int i = 0; i < alpha.n_rows; i++) {
		if (alpha[i] > 0) {
			double test = beta[i] / alpha[i];
			if (test < min) {
				min = test;
				ind = i;
            }
		}
	}

	row = ind;
	t = min;
}

void change_base(mat& A, mat& d, int row, int col, 
		vector<int>& B_index, vector<int>& nonB_index) {
	double pivot = A(row, col);
	
	// Gauss-Jordan elimination
	for (int i = 0; i < A.n_rows; i++) {
		if (i == row) continue;

		double ratio = -(A(i, col) / pivot);
		A.row(i) += A.row(row) * ratio;
	}

	double ratio = -(d[col] / pivot);
	d += A.row(row) * ratio;

	A.row(row).transform( [&](double val) { return (val / pivot); } );

	// Change base
	int aux = B_index[row];
	B_index[row] = col;
	replace(nonB_index.begin(), nonB_index.end(), col, aux);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <mps_file>" << endl;
        return EXIT_FAILURE;
	}

    mat A;
    colvec b;
    rowvec c;

    LinearProgram linear_program;
    const string& file_name = argv[1];
    if (MPSToMatrix(linear_program, file_name, A, b, c)) {
        return EXIT_FAILURE;
	}

	A.print("A:");
	
	vector<int> B_index;
	int ind;
	for (int i = 0; i < A.n_rows; ++i) {
		ind = -1;
		while ((ind < 0) || (ind >= A.n_cols)) {
			cout << "Provide base index nr. " << i+1 << ": ";
			cin >> ind;
		}
		B_index.push_back(ind);
	}

	vector<int> full;
	for (int i = 0; i < A.n_cols; i++) {
		full.push_back(i);
	} 

	vector<int> nonB_index;
	set_difference(full.begin(), full.end(), B_index.begin(), B_index.end(),
	inserter(nonB_index, nonB_index.end()));

	// Calculate xB
	mat B(A.n_rows, 0);
	for (int ind: B_index) {
		B = join_rows(B, A.col(ind));
	}

	/*
	 * sum(y_j * A^j) = b
	 * B * xB = b 
	 * *B^-1 / B * xB = b
	 * xB = B^-1 * b
	 */ 
	mat xB;
	try
	{
		xB = inv(B) * b;
	}
	catch (exception& e)
	{
		cerr << "Couldn't provide inverse for base!" << endl;
		return 1;
	}

	// Base is admissible if all elements of xB are non-negative
	// => inadmissible if at least one element is negative
	bool inadmissible = any(vectorise(xB) < 0);
	if (inadmissible) {
		xB.print("xB:");
		cerr << "Base is inadmissible, exiting!" << endl;
		return 2;
	}

	// Calculate initial z
	rowvec final_xB(A.n_cols);
	int i = 0;
	for (int ind: B_index) {
		final_xB[ind] = xB[i];
		++i;
	}
	final_xB.print("xB:");
	double z = sum(c % final_xB);
	cout << "Initial Z = " << z << endl;

	rowvec cB(B.n_rows);
	i = 0;
	for (int ind: B_index) {
		cB[i] = c[ind]; 
		i++;
	}
	cB.print("cB:");

	// Calculate simplex multiplier
	mat multi = cB * inv(B);
	multi.print("piT: ");

	// Calculate initial dT
	mat d = c - multi * A;
	d.print("Initial d:");
	d.insert_cols(A.n_cols, 1);

	int row;
	double t;
	A = join_rows(A, xB);
	A.print("A:");	

	while (true) {
		// Find the index of the minimum non-base value
		double min = 0;
		int q = -1;

		for (int ind: nonB_index) {
			if (d[ind] < min) {
				min = d[ind];
				q = ind;
			}
		}
		cout << "nonB_index: " << endl;
		for (int ind: nonB_index) {
			cout << ind << " ";
		}
		cout << endl;
		d.print("dT:");
		cout << min << " " << q << endl;

		if ((q >= 0) && (min < 0)) {
			find_pivot(A, B_index, q, row, t);
		
			if (row < 0) {
				cout << "Theta = +inf, the solution is unbounded, terminating..." << endl;
				return 1;
			}
			cout << "Pivot (" << row << " " << q << "): " << A(row, q) << endl;

			// Update target function value
			z += t * d[q];
			cout << "Z = " << z << endl;

			change_base(A, d, row, q, B_index, nonB_index);
		}
		else {
			// All non-base dT values respect their optimality condition, 
			// we have a solution!
			print_solution(A, d, B_index, z);
			break;
		}
	}

    return EXIT_SUCCESS;
}