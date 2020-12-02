class Matrix:
    class InsertionError(Exception):
        pass

    def __init__(self, n, m):
        self.rows = n
        self.columns = m
        self.matrix = []  # it will be a list of rows where each row is a list of cells (int/float)

    def __add__(self, other):
        """returns self matrix + other matrix"""
        assert self.rows == other.rows and self.columns == other.columns  # check if correct dimensions
        result = Matrix(self.rows, self.columns)
        for row1, row2 in zip(self.matrix, other.matrix):
            row = []
            for cell1, cell2 in zip(row1, row2):
                row.append(cell1 + cell2)
            result.matrix.append(row)
        return result

    def __str__(self):
        """returns a string representation of the matrix to be printed out"""
        return "\n".join([" ".join([str(round(x, 3)) for x in row]) for row in self.matrix])

    def read(self):
        """reads and fills a matrix from the input"""
        for _ in range(self.rows):
            row = [float(x) if "." in x else int(x) for x in input().split()]  # read a full row
            if len(row) != self.columns:
                raise self.InsertionError  # incorrect input
            self.matrix.append(row)

    def mul_by_num(self, sca):
        """returns matrix * a constant"""
        result = Matrix(self.rows, self.columns)
        for row in self.matrix:
            result.matrix.append([sca * x for x in row])
        return result

    def __mul__(self, other):
        """returns matrix * matrix"""
        assert self.columns == other.rows  # check if correct dimensions
        result = Matrix(self.rows, other.columns)
        for i in range(self.rows):
            row = []
            for j in range(other.columns):
                row.append(dot_product(self.matrix[i], other.column(j)))
            result.matrix.append(row)
        return result

    def column(self, index):
        """returns the column at the specified index [0, len["""
        return [x[index] for x in self.matrix]

    def trans_main_diago(self):
        """returns the transposition following the main diagonal of the matrix"""
        result = Matrix(self.columns, self.rows)
        for i in range(self.columns):
            result.matrix.append(self.column(i))  # the column of the old matrix becomes the row in the new one
        return result

    def trans_sec_diago(self):
        """returns the transposition following the secondary diagonal of the matrix"""
        result = Matrix(self.columns, self.rows)
        for i in range(self.columns - 1, -1, -1):
            result.matrix.append(list(reversed(self.column(i))))  # the last column reversed becomes the first row
        return result

    def trans_vert_line(self):
        """returns the transposition following the vertical line at the middle of the matrix"""
        result = Matrix(self.rows, self.columns)
        for row in self.matrix:
            result.matrix.append(list(reversed(row)))  # rows become reversed
        return result

    def trans_hor_line(self):
        """returns the transposition following the horizontal line at the middle of the matrix"""
        result = Matrix(self.rows, self.columns)
        for row in list(reversed(self.matrix)):
            result.matrix.append(row)  # the last row becomes the first
        return result

    def determinant(self):
        """returns the determinant of any square matrix"""
        assert self.rows == self.columns  # check if the matrix is square
        if self.rows == 1:
            return self.matrix[0][0]
        elif self.rows == 2:
            return determinant_2x2(self.matrix)
        else:
            # we'll choose the first row as our reference and ignore the 0s for faster calculations
            return sum([((-1) ** (2 + j)) * x * sub_matrix(self, 0, j).determinant() for j, x in enumerate(self.matrix[0]) if x != 0])

    def inverse(self):
        """returns the inverse matrix"""
        det = self.determinant()
        assert det != 0  # check if the inverse matrix exist
        result = Matrix(self.rows, self.columns)
        # calculate the cofactor matrix of each element of the old matrix
        for i in range(self.rows):
            row = []
            for j in range(self.columns):
                row.append(((-1) ** (i + j)) * sub_matrix(self, i, j).determinant())
            result.matrix.append(row)
        return result.trans_main_diago().mul_by_num((1 / det))  # return the transposed multiplied by the inverse of the determinant


def dot_product(row, column):
    """returns the result of the dot product of two equally length lists (a row and a column)"""
    return sum([row[index] * column[index] for index in range(len(row))])


def determinant_2x2(matrix):
    """returns the determinant of a 2x2 matrix"""
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def sub_matrix(matrix_obj, row_i, column_j):
    """returns the sub matrix of matrix_obj after we delete the row_i'th row and the column_j'th column from it, row_i and column_j [0, len["""
    result = Matrix(matrix_obj.rows - 1, matrix_obj.columns - 1)
    for i in range(matrix_obj.rows):
        if i == row_i:  # skip the deleted row
            continue
        row = []
        for j in range(matrix_obj.columns):
            if j == column_j:  # skip the cell of the deleted column
                continue
            row.append(matrix_obj.matrix[i][j])
        result.matrix.append(row)
    return result


# main program
while True:
    try:
        print("1. Add matrices\n2. Multiply matrix by a constant\n3. Multiply matrices\n4. Transpose matrix\n5. Calculate a determinant\n6. Inverse matrix\n0. Exit")
        in_put = input("Your choice: ")
        if in_put == "1":  # add two matrices
            a_row, a_column = input("Enter size of first matrix: ").split()
            A = Matrix(int(a_row), int(a_column))
            print("Enter first matrix:")
            A.read()
            b_row, b_column = input("Enter size of second matrix: ").split()
            B = Matrix(int(b_row), int(b_column))
            print("Enter second matrix:")
            B.read()
            print("The result is:")
            print(A + B)
        elif in_put == "2":  # multiply matrix by a constant
            a_row, a_column = input("Enter size of matrix: ").split()
            A = Matrix(int(a_row), int(a_column))
            print("Enter matrix:")
            A.read()
            c = input("Enter constant: ")
            c = float(c) if "." in c else int(c)
            print("The result is:")
            print(A.mul_by_num(c))
        elif in_put == "3":  # multiply two matrices
            a_row, a_column = input("Enter size of first matrix: ").split()
            A = Matrix(int(a_row), int(a_column))
            print("Enter first matrix:")
            A.read()
            b_row, b_column = input("Enter size of second matrix: ").split()
            B = Matrix(int(b_row), int(b_column))
            print("Enter second matrix:")
            B.read()
            print("The result is:")
            print(A * B)
        elif in_put == "4":  # transpose a matrix
            print("\n1. Main diagonal\n2. Side diagonal\n3. Vertical line\n4. Horizontal line")
            in_put = input("Your choice: ")
            a_row, a_column = input("Enter size of matrix: ").split()
            A = Matrix(int(a_row), int(a_column))
            print("Enter matrix:")
            A.read()
            print("The result is:")
            if in_put == "1":
                print(A.trans_main_diago())
            elif in_put == "2":
                print(A.trans_sec_diago())
            elif in_put == "3":
                print(A.trans_vert_line())
            elif in_put == "4":
                print(A.trans_hor_line())
        elif in_put == "5":  # calculate the determinant of a matrix
            a_row, a_column = input("Enter size of matrix: ").split()
            A = Matrix(int(a_row), int(a_column))
            print("Enter matrix:")
            A.read()
            print("The result is:")
            print(A.determinant())
        elif in_put == "6":  # inverse of a matrix
            try:
                a_row, a_column = input("Enter size of matrix: ").split()
                A = Matrix(int(a_row), int(a_column))
                print("Enter matrix:")
                A.read()
                print(f"The result is:\n{A.inverse()}")
            except AssertionError:
                print("This matrix doesn't have an inverse.")
        elif in_put == "0":  # exit
            break
    except AssertionError:
        print("The operation cannot be performed.")
    except (Matrix.InsertionError, ValueError):
        print("WRONG INPUT")
    print("\n")
