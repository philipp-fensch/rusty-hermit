extern crate rpc_lib;
use rpc_lib::include_rpcl;

use std::mem::size_of;

#[cfg(target_os = "hermit")]
extern crate hermit_sys;

#[include_rpcl("examples/hermit-cuda/rpc_cuda.x")]
struct RPCConnection;

fn main() {
    cuda_solve_linear_system();
}

fn cuda_solve_linear_system() {
    const DIM: usize = 3;
    // System to Solve: (transposed)
    // | 2 2 0 |       | 2 |               |-1 |
    // | 0 2 0 | * X = | 4 | Solution: X = | 2 |
    // | 0 1 1 |       | 5 |               | 3 |
    let matrix_host: [f64; DIM * DIM] = [
        2.0, 0.0, 0.0, // transposed
        2.0, 2.0, 1.0,
        0.0, 0.0, 1.0
    ];
    let right_side_host: [f64; DIM] = [
        2.0,
        4.0,
        5.0
    ];

    // Cast f64 to u8 for cudamemcpy
    let matrix_host_cast = unsafe {
        std::mem::transmute::<[f64; DIM * DIM], [u8; size_of::<f64>() * DIM * DIM]>(matrix_host)
    };

    // Init Connection and CUDA
    let rpc_connection = RPCConnection::new("137.226.133.199");
    let solver = match rpc_connection.rpc_cusolverDnCreate() { ptr_result::Case0 { ptr } => ptr, ptr_result::CaseDefault => panic!("cusolverDnCreate failed"), };

    // Allocate Memory
    let (vector_device, matrix_device, piv_seq_device, err, workspace) = allocate_memory(&rpc_connection, solver);

    // Copy Matrix to Device
    rpc_connection.CUDA_MEMCPY_HTOD(matrix_device, matrix_host_cast.to_vec(), matrix_host_cast.len() as u64);

    // LU
    lu_factorization(&rpc_connection, solver, matrix_device, workspace, piv_seq_device, err);
    rpc_connection.CUDA_DEVICE_SYNCHRONIZE();

    // Solve System
    // Copy rhs to Device
    let right_side_host_cast = unsafe {
        std::mem::transmute::<[f64; DIM], [u8; size_of::<f64>() * DIM]>(right_side_host)
    };
    rpc_connection.CUDA_MEMCPY_HTOD(vector_device, right_side_host_cast.to_vec(), right_side_host_cast.len() as u64);

    // Solve
    let res = rpc_connection.rpc_cusolverDnDgetrs(
        solver,
        0, // CUBLAS_OP_N
        3,
        1, // #right-hand-sides
        matrix_device,
        3,
        piv_seq_device,
        vector_device,
        3,
        err
    );
    rpc_connection.CUDA_DEVICE_SYNCHRONIZE();
    assert!(res == 0, "Solving System failed (cusolverDnDgetrs)");

    // Copy left-hand-side back
    let res = match rpc_connection.CUDA_MEMCPY_DTOH(vector_device, (size_of::<f64>() * DIM) as u64) { mem_result::Case0 { data } => data, mem_result::CaseDefault => panic!(""), };
    let res2 = res.as_slice();

    // Cast mem_result from generic u8 to the actual f64
    let solution = unsafe {
        std::mem::transmute::<&[u8], &[f64]>(&res2)
    };

    // Check Result
    assert!(
        (solution[0] + 1.0).abs() < 0.001 ||
        (solution[1] - 2.0).abs() < 0.001 ||
        (solution[2] - 3.0).abs() < 0.001,
        "Solution wrong"
    );

    // Free Memory
    rpc_connection.CUDA_FREE(vector_device);
    rpc_connection.CUDA_FREE(matrix_device);
    rpc_connection.CUDA_FREE(piv_seq_device);
    rpc_connection.CUDA_FREE(err);
    rpc_connection.CUDA_FREE(workspace);

    assert!(rpc_connection.rpc_cusolverDnDestroy(solver) == 0, "rpc_cusolverdndestroy failed");
}

fn allocate_memory(rpc_connection: &RPCConnection, solver: u64) -> (u64, u64, u64, u64, u64) {
    // Vector
    let rhs_vector = match rpc_connection.CUDA_MALLOC(3 * size_of::<f64>() as u64) { ptr_result::Case0 { ptr } => ptr, ptr_result::CaseDefault => panic!(""), };
    // Matrix
    let mat = match rpc_connection.CUDA_MALLOC(3 * 3 * size_of::<f64>() as u64) { ptr_result::Case0 { ptr } => ptr, ptr_result::CaseDefault => panic!(""), };
    // Pivot
    let piv = match rpc_connection.CUDA_MALLOC(3 * size_of::<f64>() as u64) { ptr_result::Case0 { ptr } => ptr, ptr_result::CaseDefault => panic!(""), };
    // Error-Code
    let err = match rpc_connection.CUDA_MALLOC(size_of::<i32>() as u64) { ptr_result::Case0 { ptr } => ptr, ptr_result::CaseDefault => panic!(""), };

    let workspace_size = match rpc_connection.rpc_cusolverDnDgetrf_bufferSize(
        solver,
        3,
        3,
        mat,
        3
    ) { int_result::Case0 { data } => data, int_result::CaseDefault => panic!(""), };
    // assert!(workspace_size == 0, "cusolverdndgetrf_buffersize failed");

    let workspace = match rpc_connection.CUDA_MALLOC(workspace_size as u64) { ptr_result::Case0 { ptr } => ptr , ptr_result::CaseDefault => panic!(""), };

    (rhs_vector, mat, piv, err, workspace)
}

fn lu_factorization(rpc_connection: &RPCConnection, solver: u64, matrix: u64, workspace: u64, piv: u64, err: u64) {
    let res = rpc_connection.rpc_cusolverDnDgetrf(
        solver,
        3,
        3,
        matrix,
        3,
        workspace,
        piv,
        err
    );
    assert!(res == 0, "LU-Factorization failed (cusolverDnDgetrf)");
}
