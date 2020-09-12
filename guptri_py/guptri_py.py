r"""
For a matrix pencil `A - λ B`, the rank-reducing `λ` are called eigenvalues.
If the pencil is regular, these are the roots of :math:`\det(A - λB)`. If the
determinant is constantly zero for all `λ`, the pencil is singular. In this
case, the eigenvalues are found by computing the GUPTRI form of the pencil.

A pencil `A - λ B` in GUPTRI form is block upper triangular such that

.. MATH::

    A - λ B =
    \begin{pmatrix}
    A_r - λ B_r & * & * & * & * \\
    0 & A_0 - λ B_0 & * & * & * \\
    0 & 0 & A_f - λ B_f & * & * \\
    0 & 0 & 0 & A_∞ - λ B_∞ & * \\
    0 & 0 & 0 & 0 & A_l - λ B_l
    \end{pmatrix}

where

* :math:`A_r - λ B_r` has all right singular structure,
* :math:`A_0 - λ B_0` has all Jordan structure of the 0 eigenvalue,
* :math:`A_f - λ B_f` has all Jordan structure for non-zero finite eigenvalues,
* :math:`A_∞ - λ B_∞` has all Jordan structure of infinite eigenvalue,
* :math:`A_l - λ B_l` has all left-singular structure of the pencil.
"""

import numpy as np

def _guptri_np(A, B, *, epsu=None, gap=1000, zero=True, part=None):
    if epsu is None:
        epsu = np.sqrt(np.finfo(np.complex128).eps)
    if not np.all(A.shape == B.shape):
        raise ValueError('matrices must be of same size')
    m, n = A.shape
    maxmn = np.max(A.shape)
    minmn = np.min(A.shape)
    wsize = (2*(maxmn*maxmn) + m*n + minmn*minmn + 12*maxmn + 3*minmn + 1)
    work = np.zeros(wsize, np.complex128)
    Ac = A.reshape(m*n, order='F').astype(np.complex128)
    Bc = B.reshape(m*n, order='F').astype(np.complex128)

    from ._fguptri_py import fguptri
    Pc, Qc, kstr, info = fguptri(Ac, Bc, m, n, epsu, gap, zero, work, wsize)
    if info:
        from warnings import warn
        warn('INFO non-zero on return from Fortran-guptri (continuing)')

    S = Ac.reshape(m, n, order='F')
    T = Bc.reshape(m, n, order='F')
    P = Pc.reshape(m, m, order='F')
    Q = Qc.reshape(n, n, order='F')

    lkstr = np.max(kstr.nonzero()[1], initial=-1) + 1
    kstr = kstr[:2,:lkstr]

    if np.isrealobj(A) and np.isrealobj(A):
        if np.all(np.isreal(S)) and np.all(np.isreal(T)):
            S = S.real
            T = T.real
        if np.all(np.isreal(P)) and np.all(np.isreal(Q)):
            P = P.real
            Q = Q.real

    kb = kcf_blocks(kstr)
    assert sum(kb[0]) == m and sum(kb[1]) == n
    if part is not None:
        part = list(part)
        end = np.cumsum(kb, 1)
        start = np.column_stack([[0, 0], end])
        rows = [k for j in part for k in range(start[0,j], end[0,j])]
        cols = [k for j in part for k in range(start[1,j], end[1,j])]
        S = S[np.ix_(rows, cols)]
        T = T[np.ix_(rows, cols)]
        P = P[:,rows]
        Q = Q[:,cols]
        kb = kb[:,part]
    return S, T, P, Q, kstr, kb

def guptri(A, B, *, epsu=None, gap=1000, zero=True,
                    subdivide=True, compact=True, part=None):
    r"""
    Compute the GUPTRI form of two matrices `A` and `B`.

    :param A: NumPy array or Sage matrix of size `m × n`
    :param B: NumPy array or Sage matrix of size `m × n`
    :param epsu: relative uncertainty in data, defaults to about `1e-8`
    :type epsu: float, optional
    :param gap: used for rank decisions in SVDs, defaults to `1000`, should be
                at least `1`
    :type gap: float, optional
    :param zero: truncate small singular values to zero during reduction
                 process, defaults to ``True``
    :type zero: boolean, optional
    :param subdivide: when using Sage, return block matrices, defaults to
                      ``True``
    :type subdivide: boolean, optional
    :param compact: strip diagonal blocks of size zero from the result when
                    using ``subdivide``, defaults to ``True``
    :type compact: boolean, optional
    :type part: list or tuple of indices `0..4`, optional
    :param part: return only the submatrices corresponding to specific diagonal
         blocks, for example

         * ``[0]`` - minimal left and right reducing subspaces
         * ``[0,1,2,3]`` - maximal left and right reducing subspaces

    :rtype: 5-tuple of NumPy arrays or Sage matrices (depending on input)
    :return: `S`, `T`, `P`, `Q`, ``kstr`` where

        * `S - λ T` is in GUPTRI form,
        * `P`, `Q` are unitary,
        * ``kstr`` contains the Kronecker structure of the pencil (see the
          References_),

        such that

        .. MATH::

            P^* (A - λ B) Q = S - λ T.

    The eigenvalues are the ratios of the diagonal elements of the regular part
    of the pencil, i.e. the three central square diagonal blocks of `S` and
    `T`.

    The leading columns of `P` and `Q` form orthogonal bases of the
    corresponding left and right reducing subspaces, respectively.

    EXAMPLES:

    Using NumPy::

        >>> import numpy as np
        >>> from guptri_py import *
        >>> A = np.array([[22,34,31,31,17],
        ...               [45,45,42,19,29],
        ...               [39,47,49,26,34],
        ...               [27,31,26,21,15],
        ...               [38,44,44,24,30]], np.float)
        >>> B = np.array([[13,26,25,17,24],
        ...               [31,46,40,26,37],
        ...               [26,40,19,25,25],
        ...               [16,25,27,14,23],
        ...               [24,35,18,21,22]], np.float)
        >>> S, T, P, Q, kstr = guptri(A, B); S, T, kstr
        (array([[   0.        ,    0.        ,  -31.21794153,   69.71856621, -142.67272727],
                [   0.        ,    0.        ,    0.        ,   15.85324665,  -42.10391654],
                [   0.        ,    0.        ,    0.        ,  -15.56393985,   -3.47120411],
                [   0.        ,    0.        ,    0.        ,    0.        ,   13.59793536],
                [   0.        ,    0.        ,    0.        ,    0.        ,    0.        ]]),
         array([[   0.        ,  -21.59418044,  -22.16602894,   44.51530913, -110.98181818],
                [   0.        ,    0.        ,  -25.05806277,   19.3190495 ,  -43.89049025],
                [   0.        ,    0.        ,    0.        ,   -7.78196993,    9.20409929],
                [   0.        ,    0.        ,    0.        ,    0.        ,    0.        ],
                [   0.        ,    0.        ,    0.        ,    0.        ,    0.        ]]),
         array([[ 2,  1,  0, -1,  2,  0, -1,  1, -1],
                [ 1,  1,  0, -1,  1,  0, -1,  1, -1]], dtype=int32))

    ::

        >>> np.linalg.norm(A - P.dot(S.dot(Q.T.conj()))) < 1e-12
        True
        >>> np.linalg.norm(B - P.dot(T.dot(Q.T.conj()))) < 1e-12
        True

    We extract the block sizes and conclude that there are two `0` eigenvalues,
    one finite eigenvalue at `2` and one infinite eigenvalue::

        >>> kcf_blocks(kstr)
        array([[0, 2, 1, 1, 1],
               [1, 2, 1, 1, 0]])
        >>> S[2,3] / T[2,3]
        2.0

    |

    Using Sage::

        sage: from guptri_py import *
        sage: A = matrix(A); B = matrix(B)
        sage: S, T, P, Q, kstr = guptri(A, B)
        sage: S  # tol 1e-13
        [-------------------+---------------------------------------+-------------------+-------------------+]
        [                0.0|                0.0 -31.217941525933004|  69.71856620722244|-142.67272727272717|]
        [                0.0|                0.0                 0.0|  15.85324664502821| -42.10391654473494|]
        [-------------------+---------------------------------------+-------------------+-------------------+]
        [                0.0|                0.0                 0.0|-15.563939851607742|-3.4712041141556753|]
        [-------------------+---------------------------------------+-------------------+-------------------+]
        [                0.0|                0.0                 0.0|                0.0| 13.597935363644218|]
        [-------------------+---------------------------------------+-------------------+-------------------+]
        [                0.0|                0.0                 0.0|                0.0|                0.0|]
        sage: T  # tol 1e-13
        [-------------------+---------------------------------------+-------------------+-------------------+]
        [                0.0|-21.594180436280595 -22.166028942864237|  44.51530913141558| -110.9818181818181|]
        [                0.0|                0.0 -25.058062773982755|  19.31904950452907| -43.89049025352817|]
        [-------------------+---------------------------------------+-------------------+-------------------+]
        [                0.0|                0.0                 0.0| -7.781969925803871|  9.204099294191801|]
        [-------------------+---------------------------------------+-------------------+-------------------+]
        [                0.0|                0.0                 0.0|                0.0|                0.0|]
        [-------------------+---------------------------------------+-------------------+-------------------+]
        [                0.0|                0.0                 0.0|                0.0|                0.0|]
        sage: P  # tol 1e-13
        [|  0.2696799449852967  0.45203748616024275|  0.8072453264583416|  0.0225657325979767|-0.26604625350141314]
        [|  0.4045199174779453   0.5100693255997332| -0.4984905633467818|  0.5399884462530116|-0.19003303821529563]
        [|  0.6741998624632421    -0.38178841736507|  0.1704780900615978|  0.2135798916315633|  0.5700991146458833]
        [| 0.13483997249264842   0.5619925503613826|-0.17710779356399256| -0.6188095967360872|  0.5016872208883796]
        [|  0.5393598899705936 -0.27183335316392954| -0.1985754049050833| -0.5285466663441797|  -0.562497793117271]
        sage: Q  # tol 1e-13
        [ -0.4043680421515943| 0.15099829724341438   0.4636322195173679|  0.7252612584539466| -0.2696799449852965|]
        [  0.6984538909891174|  -0.409711992194672   0.1200055765268443| 0.19745332690892747| -0.5393598899705937|]
        [  0.1470429244187616|  0.6895752857826296   -0.563291481656615| 0.14808999518169602| -0.4045199174779448|]
        [ -0.4043680421515946|-0.03098662125842025  0.18594270648664857| -0.5885628013631496| -0.6741998624632417|]
        [-0.40436804215159394| -0.5769413767639264  -0.6471258326055097|  0.2582081967270594|-0.13483997249264815|]
        sage: kstr
        [ 2  1  0 -1  2  0 -1  1 -1]
        [ 1  1  0 -1  1  0 -1  1 -1]
        sage: (A - P * S * Q.H).norm() < 1e-12
        True
        sage: (B - P * T * Q.H).norm() < 1e-12
        True

    A rectangular example::

        sage: A = matrix([[0,1,0], [0,0,2]])
        sage: B = matrix([[0,0,0], [0,0,3]])
        sage: guptri(A, B)
        (
        [----+----+----]  [---+---+---]
        [ 0.0| 2.0| 0.0]  [0.0|3.0|0.0]              [-1.0| 0.0|-0.0]
        [----+----+----]  [---+---+---]  [|0.0|1.0]  [-0.0| 0.0|-1.0]
        [ 0.0| 0.0|-1.0], [0.0|0.0|0.0], [|1.0|0.0], [ 0.0| 1.0|-0.0],
        <BLANKLINE>
        [ 1 -1  1  0 -1  1 -1]
        [ 0 -1  1  0 -1  1 -1]
        )

    Preserving empty diagonal blocks::

        sage: S, T = guptri(A, B, compact=False)[:2]; S, T
        (
        [----++----+----+]  [---++---+---+]
        [----++----+----+]  [---++---+---+]
        [ 0.0|| 2.0| 0.0|]  [0.0||3.0|0.0|]
        [----++----+----+]  [---++---+---+]
        [ 0.0|| 0.0|-1.0|]  [0.0||0.0|0.0|]
        [----++----+----+], [---++---+---+]
        )
        sage: S.subdivision(3, 3), T.subdivision(3, 3)
        ([-1.0], [0.0])

    An example showing that SciPy, using LAPACK, is not suited for solving
    singular eigenvalue problems::

        sage: A = matrix(RDF, [[1, 2e-16], [3e-16, 0]])
        sage: B = matrix(RDF, [[1, 1e-16], [1e-16, 0]])
        sage: import scipy.linalg
        sage: scipy.linalg.eigvals(A, B)
        array([2.+0.j, 3.+0.j])
        sage: scipy.linalg.eigvals(A, B, homogeneous_eigvals=True)
        array([[2.e-16+0.j, 3.e-16+0.j],
               [1.e-16+0.j, 1.e-16+0.j]])

    Using GUPTRI instead, we find a single eigenvalue at `1`::

        sage: guptri(A, B)[:2]
        (
        [---+---+]  [---+---+]
        [0.0|1.0|]  [0.0|1.0|]
        [---+---+]  [---+---+]
        [0.0|0.0|], [0.0|0.0|]
        )

    The minimal and maximal left and right reducing subspaces::

        sage: guptri(A, B, part=[0])[2:4]
        (
            [0.0]
        [], [1.0]
        )
        sage: guptri(A, B, part=range(4))[2:4]
        (
        [|  1.0]  [0.0|1.0]
        [|1e-16], [1.0|0.0]
        )

    TESTS::

        sage: import guptri_py
        sage: TestSuite(guptri_py.guptri_py._tests_sage()).run(skip='_test_pickling')
    """
    if isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
        return _guptri_np(A, B, epsu=epsu, gap=gap, zero=zero,
                          part=part)[:-1]
    # otherwise assume A, B are Sage matrices
    S, T, P, Q, kstr, kb = \
        _guptri_np(A.numpy(), B.numpy(), epsu=epsu, gap=gap, zero=zero,
                   part=part)

    from sage.all import matrix, ZZ
    S = matrix(np.ascontiguousarray(S))
    T = matrix(np.ascontiguousarray(T))
    P = matrix(np.ascontiguousarray(P))
    Q = matrix(np.ascontiguousarray(Q))

    if subdivide:
        rows, cols = kb
        if compact:
            # skip 0×0 blocks
            rows, cols = ([rows[j] for j in range(len(rows)) if rows[j] or cols[j]],
                          [cols[j] for j in range(len(rows)) if rows[j] or cols[j]])
        rows = list(np.cumsum(rows)[:-1])
        cols = list(np.cumsum(cols)[:-1])

        S.subdivide(row_lines=rows, col_lines=cols)
        T.subdivide(row_lines=rows, col_lines=cols)
        P.subdivide(col_lines=rows)
        Q.subdivide(col_lines=cols)

    return S, T, P, Q, matrix(ZZ, np.ascontiguousarray(kstr))


def kcf_blocks(kstr):
    """
    Compute the block sizes in the Kroncker structure ``kstr`` returned by
    :func:`guptri`.

    :param kstr: the Kronecker structure computed by :func:`guptri`
    :type kstr: NumPy array or Sage matrix
    :return: NumPy array of size `2 × 5` where index `(0,k)` and `(1,k)` are
        the number of rows and columns of diagonal block `k`, respectively

    EXAMPLES::

        >>> import numpy as np
        >>> from guptri_py import *
        >>> A = np.array([[0,1,0], [0,0,2]], np.float)
        >>> B = np.array([[0,0,0], [0,0,3]], np.float)
        >>> kstr = guptri(A, B)[-1]
        >>> kcf_blocks(kstr)
        array([[0, 0, 1, 1, 0],
               [1, 0, 1, 1, 0]])
    """
    if not isinstance(kstr, np.ndarray):
        # assume that kstr is a Sage matrix
        kstr = kstr.numpy()
    kblocks = np.nonzero(kstr < 0)[1]  # find (-1)-columns in kstr
    assert len(kblocks) == 6 and np.all(kblocks[:3] == kblocks[3:])
    b0, b1, b2 = kblocks[:3]
    assert b2 + 1 == kstr.shape[1]  # b2 is last column
    assert b2 - b1 <= 2  # last block has length ≤ 1
    B0 = kstr[:,:b0]
    B1 = kstr[:,b0+1:b1]
    B2 = kstr[:,b1+1:b2]

    rows, cols = [], []
    # Lj blocks
    rows.append(sum((B0[0,j] - B0[1,j]) * j for j in range(B0.shape[1])))
    cols.append(sum((B0[0,j] - B0[1,j]) * (j+1) for j in range(B0.shape[1])))
    # Jj(0) blocks
    s = sum((B0[1,j-1] - B0[0,j]) * j for j in range(1, B0.shape[1]))
    rows.append(s)
    cols.append(s)
    # regular block
    rows.append(B2[0,0] if B2.shape[1] > 0 else 0)
    cols.append(B2[1,0] if B2.shape[1] > 0 else 0)
    # Nj blocks
    s = sum((B1[1,j-1] - B1[0,j]) * j for j in range(1, B1.shape[1]))
    rows.append(s)
    cols.append(s)
    # Lj' blocks
    rows.append(sum((B1[0,j] - B1[1,j]) * (j+1) for j in range(B1.shape[1])))
    cols.append(sum((B1[0,j] - B1[1,j]) * j for j in range(B1.shape[1])))
    return np.array([rows, cols])


def _tests_sage():
    from sage.all import SageObject, matrix, RDF, CDF
    from numpy.linalg import matrix_rank

    class GuptriTests(SageObject):

        def check_guptri_properties(self, A, B):
            S, T, P, Q, kstr = guptri(A, B)
            tol = 1e-12
            assert (P.H * A * Q - S).norm() < tol
            assert (P.H * B * Q - T).norm() < tol
            assert (P.H * P - matrix.identity(RDF, P.ncols())).norm() < tol
            assert (Q.H * Q - matrix.identity(RDF, Q.ncols())).norm() < tol
            kb = kcf_blocks(kstr)
            assert np.all(kb[:,0] == 0) or kb[0,0] < kb[1,0]
            assert np.all(kb[:,-1] == 0) or kb[0,-1] > kb[1,-1]
            assert np.all(kb[0,1:4] == kb[1,1:4])

            # test that Y = A X + B X for some of the reducing subspaces
            for k in range(1, 5):
                Y, X = guptri(A, B, part=range(k))[2:4]
                AXBX = (A * X).augment(B * X)
                assert Y.ncols() == matrix_rank(AXBX, tol=1e-12)
                assert Y.ncols() == matrix_rank(AXBX.augment(Y), tol=1e-12)

        def _test_1(self, **kwds):
            A = matrix(RDF, [[0,1,0], [0,0,2]])
            B = matrix(RDF, [[0,0,0], [0,0,3]])
            self.check_guptri_properties(A, B)

        def _test_2(self, **kwds):
            A = np.array([[22,34,31,31,17],
                          [45,45,42,19,29],
                          [39,47,49,26,34],
                          [27,31,26,21,15],
                          [38,44,44,24,30]], np.float)
            B = np.array([[13,26,25,17,24],
                          [31,46,40,26,37],
                          [26,40,19,25,25],
                          [16,25,27,14,23],
                          [24,35,18,21,22]], np.float)
            self.check_guptri_properties(matrix(A), matrix(B))

        def _test_3(self, **kwds):
            A = matrix(CDF, [[1+1j, 3e-16j], [2e-16j, 0]])
            B = matrix(CDF, [[1, 1e-16j], [1e-16, 0]])
            self.check_guptri_properties(A, B)

    return GuptriTests()
