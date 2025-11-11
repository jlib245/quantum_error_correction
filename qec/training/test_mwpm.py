"""
Test MWPM decoder standalone
"""
import argparse
from qec.core.codes import Get_surface_Code
from qec.decoders.mwpm import MWPM_Decoder


def main(args):
    print(f"\n--- L={args.L}, p_error={args.p_error}, y_ratio={args.y_ratio} MWPM Test ---")
    print(f"Test shots: {args.n_test_shots}\n")

    # Load code
    Hx, Hz, Lx, Lz = Get_surface_Code(args.L)
    print(f"L{args.L} Surface Code loaded (n_qubits: {Hx.shape[1]})")

    # Create decoder
    decoder = MWPM_Decoder(Hx, Hz, Lx, Lz)

    # Evaluate
    results = decoder.evaluate(
        p_error=args.p_error,
        n_shots=args.n_test_shots,
        y_ratio=args.y_ratio,
        verbose=True
    )

    # Print results
    print("\n--- MWPM Decoder Results ---")
    print(f"Logical Error Rate (LER): {results['ler']:.8f}")
    print(f"Logical Errors: {results['logical_errors']} / {results['total_shots']}")
    print(f"Average Decoding Time: {results['avg_latency']:.6f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MWPM Decoder Test")
    parser.add_argument('-L', type=int, default=3, help='Code distance')
    parser.add_argument('-p', '--p_error', type=float, default=0.09,
                        help='Physical error rate')
    parser.add_argument('-n', '--n_test_shots', type=int, default=10000,
                        help='Number of test shots')
    parser.add_argument('-y', '--y_ratio', type=float, default=0.0,
                        help='Y-error ratio for correlated noise (0.0 = independent)')

    args = parser.parse_args()
    main(args)
