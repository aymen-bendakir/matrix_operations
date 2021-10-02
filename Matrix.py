import numpy as np
from collections import Counter
from itertools import chain, combinations


class Matrix:
    """represents a mathematical matrix with n rows and m columns"""
    class InputMatrixException(Exception):
        pass

    class IncompatibleMatricesException(Exception):
        pass

    class NonInversibleMatrixException(Exception):
        pass

    def __init__(self, n, m, body):
        self.rows = n  # number of rows
        self.columns = m  # number of columns
        self.body = body  # the body of the matrix (array of n arrays of m cells)

    @classmethod
    def Matrix(cls, n, m, body):
        return cls()

    @staticmethod
    def read(n, m):
        """reads the body of a n*m matrix from the input"""
        print(f"Enter {n} matrix rows where each row has {m} elements seperated by a single space")
        body = []
        for i in range(n):  # read n rows from input
            body.append(tuple(map(float, input(f"row {i + 1}: ").strip().split())))
            if len(body[-1]) != m:  # if the row doesn't have m cells
                raise Matrix.InputMatrixException
        return np.array(body)

    def get_row(self, i):
        """:returns the row i of the matrix"""
        return self.body[i]

    def get_column(self, j):
        """:returns the column j of the matrix"""
        return self.body[:, j]

    def __str__(self):
        max_ = max(max(len(str(x)) for x in row) for row in self.body)  # max length representation of a cell
        def represent(x):
            if not x:
                x = 0.0
            i = max_ - len(str(x))
            if i % 2 == 0:
                return str(x).join([" " * (i // 2)] * 2)
            return str(x).join([" " * (i // 2 + 1), " " * (i // 2)])
        return f"Matrix {self.rows}x{self.columns}:\n( " + " )\n( ".join(" ".join(represent(x) for x in row) for row in self.body) + " )"

    def __add__(self, other):
        if self.rows != other.rows or self.columns != other.columns:
            raise Matrix.IncompatibleMatricesException
        body = [[self.body[i][j] + other.body[i][j] for j in range(self.columns)] for i in range(self.rows)]
        return Matrix(self.rows, self.columns, np.array(body))

    def __mul__(self, other):
        if type(other) != Matrix:  # if it's a multiplication by a scalar
            return Matrix(self.rows, self.columns, np.array(self.body * other))
        # it's a multiplication by another matrix
        if self.columns != other.rows:
            raise Matrix.IncompatibleMatricesException
        def dot_product(row, column):
            """:returns the dot product of two equaly length vectors"""
            return sum(row[i] * column[i] for i in range(len(row)))
        body = [[dot_product(self.get_row(i), other.get_column(j)) for j in range(other.columns)] for i in range(self.rows)]
        return Matrix(self.rows, other.columns, np.array(body))
    
    def transpose(self):
        """:returns the transpose of the matrix following the main diagonal"""
        body = [self.get_column(j) for j in range(self.columns)]
        return Matrix(self.columns, self.rows, np.array(body))

    def minor_matrix(self, i_, j_):
        """:returns the minor matrix of the i and j element"""
        body = [[self.body[i][j] for j in range(self.columns) if j != j_] for i in range(self.rows) if i != i_]
        return Matrix(self.rows - 1, self.columns - 1, np.array(body))

    def determinant(self):
        """:returns the determinant of a square matrix, launches exception if the matrix isn't square"""
        if self.rows != self.columns:
            raise Matrix.IncompatibleMatricesException
        if self.rows == 1:
            return self.body[0][0]
        elif self.rows == 2:
            return self.body[0][0] * self.body[1][1] - self.body[1][0] * self.body[0][1]
        else:
            zero_count = [Counter(row)[0.0] for row in chain(self.body, [self.get_column(j) for j in range(self.columns)])]
            index = zero_count.index(max(zero_count))  # find the index of the row/column with most 0
            if index < self.rows:  # we're gonna go through a row
                return sum((-1) ** (index + j) * self.body[index][j] * self.minor_matrix(index, j).determinant() for j in range(self.columns) if self.body[index][j])
            index -= self.rows  # we're gonna go through a column
            return sum((-1) ** (i + index) * self.body[i][index] * self.minor_matrix(i, index).determinant() for i in range(self.rows) if self.body[i][index])

    def inverse(self):
        """:returns the inverse of a square matrix if it's possible"""
        det = self.determinant()
        if det == 0:
            raise Matrix.NonInversibleMatrixException
        det = 1 / det
        body = [[(-1) ** (i + j) * self.minor_matrix(i, j).determinant() * det for i in range(self.rows)] for j in range(self.columns)]
        return Matrix(self.rows, self.columns, np.array(body))

    def rank(self):
        """:returns a rank object of the matrix"""
        rank = min(self.rows, self.columns)
        for r in range(rank, 0, -1):
            for row_comb in map(list, combinations(range(self.rows), r)):
                body = self.body[row_comb]
                for column_comb in map(list, combinations(range(self.columns), r)):
                    m = Matrix(r, r, np.array(body[:, column_comb]))
                    det = m.determinant()
                    if det != 0:
                        return Rank(self, r, m, det, [row_comb, column_comb])
        return "Rank: 0"


class Rank:
    def __init__(self, matrix, r, significant_matrix, significant_det, list_i_j):
        self.main_matrix = matrix  # the old matrix we wanted to calculate its rank
        self.rank = r  # the rank
        self.sub_matrix = significant_matrix  # the significant sub matrix used for calculation the rank
        self.sub_det = significant_det  # the determinant of the sub matrix
        self.i_j = list_i_j  # list of row and column indexes of the sub_matrix: list([i], [j])
    
    def __str__(self):
        return "Rank: " + str(self.rank) + "\nSub " + str(self.sub_matrix) + "\nIts Determinant: " + str(self.sub_det)

    def __repr__(self):
        return "Rank: " + str(self.rank) + "\nSub " + str(self.sub_matrix) + "\nIts Determinant: " + str(self.sub_det) + "\nList of Indexes: " + str(self.i_j)


if __name__ == '__main__':
    while True:
        try:
            print("1. Add matrices\n2. Multiply matrix by a scalar\n3. Multiply matrices\n4. Transpose matrix\n5. Calculate determinant of square matrix\n6. Calculate rank\n7. Inverse a square matrix\n0. Exit")
            in_put = input("Your choice: ")
            print()
            if in_put == "1":  # add two matrices
                a_row, a_column = map(int, input("Enter size of first matrix (n m): ").split())
                A = Matrix(a_row, a_column, Matrix.read(a_row, a_column))
                b_row, b_column = map(int, input("Enter size of second matrix (n m): ").split())
                B = Matrix(b_row, b_column, Matrix.read(b_row, b_column))
                print("\nThe result is a " + str(A + B))
            elif in_put == "2":  # multiply matrix by a constant
                a_row, a_column = map(int, input("Enter size of the matrix (n m): ").split())
                A = Matrix(a_row, a_column, Matrix.read(a_row, a_column))
                c = float(input("Enter the scalar: "))
                print("\nThe result is a " + str(A * c))
            elif in_put == "3":  # multiply two matrices
                a_row, a_column = map(int, input("Enter size of first matrix (n m): ").split())
                A = Matrix(a_row, a_column, Matrix.read(a_row, a_column))
                b_row, b_column = map(int, input("Enter size of second matrix (n m): ").split())
                B = Matrix(b_row, b_column, Matrix.read(b_row, b_column))
                print("\nThe result is a " + str(A * B))
            elif in_put == "4":  # transpose a matrix
                a_row, a_column = map(int, input("Enter size of the matrix (n m): ").split())
                A = Matrix(a_row, a_column, Matrix.read(a_row, a_column))
                print("\nThe result is a " + str(A.transpose()))
            elif in_put == "5":  # calculate the determinant of a square matrix
                a_row = int(input("Enter size of the square matrix (n): "))
                A = Matrix(a_row, a_row, Matrix.read(a_row, a_row))
                print("\nThe determinant: " + str(A.determinant()))
            elif in_put == "6":  # calculate the rang of a matrix
                a_row, a_column = map(int, input("Enter size of the matrix (n m): ").split())
                A = Matrix(a_row, a_column, Matrix.read(a_row, a_column))
                print("\nThe results are:\n" + str(A.rank()))
            elif in_put == "7":  # inverse of a matrix
                try:
                    a_row = int(input("Enter size of the square matrix (n): "))
                    A = Matrix(a_row, a_row, Matrix.read(a_row, a_row))
                    print("\nThe inverse is a " + str(A.inverse()))
                except Matrix.NonInversibleMatrixException:
                    print("\nThis matrix doesn't have an inverse.")
            elif in_put == "0":  # exit
                break
        except Matrix.IncompatibleMatricesException:
            print("\nThis operation cannot be performed, wrong matrix sizes.")
        except (Matrix.InputMatrixException, ValueError):
            print("\nWRONG INPUT. Please follow the input format.")
        print("\n")
