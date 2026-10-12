/* eigen_symmetric.h — GPU-callable eigendecomposition for small symmetric matrices.
 *
 * Replaces GSL gsl_eigen_symmv / gsl_eigen_symm for 1x1, 2x2, and 3x3 cases.
 * Uses cyclic Jacobi rotations. Every rotation is orthogonal, so the eigenvectors
 * are orthonormal by construction, including repeated and nearly repeated
 * eigenvalues, and eigenvalues are accurate to roundoff relative to the matrix
 * norm (the same accuracy class as GSL). A 2x2 matrix is diagonalized exactly by
 * its single rotation. The number of sweeps is bounded, so the cost on a device
 * is predictable; no workspace is allocated.
 *
 * Reference: H. Rutishauser, "The Jacobi method for real symmetric matrices",
 *            Numer. Math. 9 (1966) 1-10 (the rotation formulas used below).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef EIGEN_SYMMETRIC_H
#define EIGEN_SYMMETRIC_H

#include <math.h>
#include <float.h>
/* under a GPU compiler fall back to host+device, as macros.h does, never to host-only */
#ifndef GIZMO_GPU_FUNCTION
#if (defined(__CUDACC__) || defined(__HIPCC__))
#define GIZMO_GPU_FUNCTION __device__ __host__
#else
#define GIZMO_GPU_FUNCTION
#endif
#endif

#define EIGEN_SYMMETRIC_MAX_SWEEPS 16 /* finite 2x2 and 3x3 input converges within 4 sweeps (including repeated and widely scaled eigenvalues); this only bounds pathological input */

/* ================================================================
   eigen_symmetric_jacobi: shared core of the two entry points below.
   A[n*n] is a symmetric matrix in row-major order; only its upper triangle is
   read, and it is not modified. eval[n] receives the eigenvalues in ascending
   order. If want_vectors, evec[n*k + j] receives component j of the unit
   eigenvector belonging to eval[k] (each eigenvector stored contiguously);
   otherwise evec is not touched and no vector work is compiled in.
   n must be 1, 2, or 3. Returns 1 when the off-diagonal part reached roundoff
   (relative to the matrix norm) within the sweep bound, 0 otherwise.
   ================================================================ */
/* put d[i] <= d[j], swapping the matching eigenvector columns */
template<bool want_vectors>
GIZMO_GPU_FUNCTION
static inline void eigen_symmetric_order_pair(double d[3], double v[3][3], int i, int j)
{
    if(!(d[i] > d[j])) {return;}
    double t = d[i]; d[i] = d[j]; d[j] = t;
    if constexpr(want_vectors) {for(int r = 0; r < 3; r++) {double s = v[r][i]; v[r][i] = v[r][j]; v[r][j] = s;}}
}

template<bool want_vectors>
GIZMO_GPU_FUNCTION
static inline int eigen_symmetric_jacobi(const double *A, double *eval, double *evec, int n)
{
    /* work on A with each entry divided by the largest |entry|, so no square or sum below can overflow or
       underflow anywhere in the finite range (subnormal entries included) */
    double scale = 0;
    int finite = 1;
    for(int i = 0; i < n; i++) {
        for(int j = i; j < n; j++) {
            double x = fabs(A[n*i + j]);
            if(!(x <= DBL_MAX)) {finite = 0;} else if(x > scale) {scale = x;}
        }
    }
    if(!finite) { /* a NaN or infinite entry has no eigensystem: NaN eigenvalues, unconverged */
        for(int k = 0; k < n; k++) {
            eval[k] = NAN;
            if constexpr(want_vectors) {for(int j = 0; j < n; j++) {evec[n*k + j] = (j == k) ? 1.0 : 0.0;}}
        }
        return 0;
    }
    if(scale == 0) {scale = 1;} /* zero matrix */
    double a[3][3], v[3][3];
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            a[i][j] = ((i <= j) ? A[n*i + j] : A[n*j + i]) / scale;
            if constexpr(want_vectors) {v[i][j] = (i == j) ? 1.0 : 0.0;}
        }
    }

    double norm2 = 0; /* squared Frobenius norm: invariant under the rotations */
    for(int i = 0; i < n; i++) {for(int j = 0; j < n; j++) {norm2 += a[i][j] * a[i][j];}}
    const double tolerance2 = DBL_EPSILON * DBL_EPSILON * norm2;

    int converged = 0;
    for(int sweep = 0; sweep <= EIGEN_SYMMETRIC_MAX_SWEEPS; sweep++) {
        double off2 = 0;
        for(int p = 0; p < n; p++) {for(int q = p+1; q < n; q++) {off2 += a[p][q] * a[p][q];}}
        if(off2 <= tolerance2) {converged = 1; break;}
        if(sweep == EIGEN_SYMMETRIC_MAX_SWEEPS) {break;}

        for(int p = 0; p < n; p++) {
            for(int q = p+1; q < n; q++) {
                double apq = a[p][q];
                if(apq == 0) {continue;}
                /* rotation angle that zeroes a[p][q]; t = tan of it, the smaller root for stability */
                double theta = (a[q][q] - a[p][p]) / (2.0 * apq);
                double t = (fabs(theta) > 1.0e150) ? 0.5 / theta : ((theta >= 0) ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta*theta + 1.0));
                double c = 1.0 / sqrt(t*t + 1.0), s = t * c, tau = s / (1.0 + c);
                a[p][p] -= t * apq;
                a[q][q] += t * apq;
                a[p][q] = a[q][p] = 0;
                for(int r = 0; r < n; r++) {
                    if(r == p || r == q) {continue;}
                    double arp = a[r][p], arq = a[r][q];
                    a[r][p] = a[p][r] = arp - s * (arq + tau * arp);
                    a[r][q] = a[q][r] = arq + s * (arp - tau * arq);
                }
                if constexpr(want_vectors) {
                    for(int r = 0; r < n; r++) {
                        double vrp = v[r][p], vrq = v[r][q];
                        v[r][p] = vrp - s * (vrq + tau * vrp);
                        v[r][q] = vrq + s * (vrp - tau * vrq);
                    }
                }
            }
        }
    }

    /* ascending order by a fixed compare-swap network (constant indices keep everything in registers on a
       device); column k of v is the eigenvector of d[k] */
    double d[3] = {a[0][0], (n > 1) ? a[1][1] : 0.0, (n > 2) ? a[2][2] : 0.0};
    if(n > 1) {eigen_symmetric_order_pair<want_vectors>(d, v, 0, 1);}
    if(n > 2) {eigen_symmetric_order_pair<want_vectors>(d, v, 1, 2); eigen_symmetric_order_pair<want_vectors>(d, v, 0, 1);}
    for(int k = 0; k < n; k++) {
        eval[k] = d[k] * scale;
        if constexpr(want_vectors) {for(int j = 0; j < n; j++) {evec[n*k + j] = v[j][k];}}
    }
    return converged;
}


/* ================================================================
   eigen_symmetric_values: eigenvalues only (no eigenvector work).
   A[n*n] is a symmetric matrix in row-major order (not modified).
   eval[n] receives the eigenvalues in ascending order.
   n must be 1, 2, or 3. Returns 1 on convergence (see above).
   ================================================================ */
GIZMO_GPU_FUNCTION
static inline int eigen_symmetric_values(const double *A, double *eval, int n)
{
    return eigen_symmetric_jacobi<false>(A, eval, (double *)0, n);
}


/* ================================================================
   eigen_symmetric: eigenvalues AND eigenvectors.
   A[n*n] is a symmetric matrix in row-major order (not modified).
   eval[n] receives the eigenvalues in ascending order.
   evec[n*n] receives the unit eigenvectors, each stored contiguously:
   evec[n*k + j] = j-th component of the eigenvector of eval[k].
   n must be 1, 2, or 3. Returns 1 on convergence (see above).
   ================================================================ */
GIZMO_GPU_FUNCTION
static inline int eigen_symmetric(const double *A, double *eval, double *evec, int n)
{
    return eigen_symmetric_jacobi<true>(A, eval, evec, n);
}


#endif /* EIGEN_SYMMETRIC_H */
