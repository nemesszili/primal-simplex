#include <iostream>
#include <exception>
#include <algorithm>
#include <vector>

#include "armadillo"

using namespace arma;
using namespace std;

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

int main(int argc, char** argv)
{
	cout << "Armadillo version: " << arma_version::as_string() << endl;
	mat A;

	A << 2  << 1 << 1 << 1 << 0 << 0 << endr
	  << 0  << 1 << 2 << 0 << 1 << 0 << endr
	  << -1 << 0 << 3 << 0 << 0 << 1 << endr;

	rowvec c;
	c << -1 << -1 << -2 << 0 << 0 << 0 << endr;

	colvec b;
	b << 1 << 3 << 2 << endr;

	vector<int> B_index;
	B_index.push_back(3);
	B_index.push_back(4);
	B_index.push_back(5);

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
	// double z = 0;
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
	// mat d = c - multi * A;
	mat d = c;
	d.print("Initial d:");
	d.insert_cols(A.n_cols, 1);

	int row;
	double t;
	// A = join_rows(A, xB);
	A = join_rows(A, b);
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
			cout << "t = " << t << endl;
			cout << "d = " << d[q] << endl;
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
	
	// TODO: read from MPS
	// TODO: convert system to matrix form

	return 0;
}
