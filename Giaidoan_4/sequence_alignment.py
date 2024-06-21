def alignment(X, Y):
    """
    Tìm kiếm sự tương đồng giữa hai chuỗi.

    Args:
        X: Một chuỗi gồm n phần tử.
        Y: Một chuỗi gồm m phần tử.

    Returns:
        Một chuỗi gồm n + m phần tử, biểu diễn sự tương đồng giữa X và Y.
    """

    n = len(X)
    m = len(Y)

    # Khởi tạo ma trận S.
    S = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

    # Duyệt qua từng phần tử trong X và Y.
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if X[i - 1] == Y[j - 1]:
                S[i][j] = S[i - 1][j - 1] + 1
            else:
                S[i][j] = max(S[i - 1][j], S[i][j - 1])

    # Tìm kiếm đường dẫn tối ưu.
    i = n
    j = m
    O = []
    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            O.append(X[i - 1])
            i -= 1
            j -= 1
        else:
            if S[i - 1][j] > S[i][j - 1]:
                i -= 1
            else:
                j -= 1

    # Sắp xếp lại các phần tử trong chuỗi O theo thứ tự ngược lại.
    O.reverse()

    return O

def alignment_multiple(X):
    """
    Tìm kiếm sự tương đồng giữa nhiều chuỗi và trả về chuỗi duy nhất biểu diễn sự tương đồng cao nhất.

    Args:
        X: Một danh sách gồm n chuỗi.

    Returns:
        Một chuỗi biểu diễn sự tương đồng cao nhất giữa các chuỗi trong X.
    """

    n = len(X)
    max_similarity = []  # Biến lưu trữ chuỗi có sự tương đồng cao nhất
    max_length = 0

    # Duyệt qua từng cặp chuỗi trong danh sách X
    for i in range(n):
        for j in range(i + 1, n):  # Chỉ duyệt từ phần tử sau i đến cuối
            aligned = alignment(X[i], X[j])  # So sánh X[i] và X[j]
            if len(aligned) > max_length:
                max_similarity = aligned
                max_length = len(aligned)

    return max_similarity