#!/bin/bash
# round15_timing.sh -- round 15 timing (vnet/PLAN.md bars F3, T1), serialised and interleaved, with cool-down pauses.
#   bash benchmarks/round15_timing.sh > results_round15_timing.log 2>&1
# Uses build/windows/sldrnc_latency.exe: <model> <stream> [repeats] [lp] [threads] [observe] [fleet robots]
cd "$(dirname "$0")/.."
L=build/windows/sldrnc_latency.exe
R=results
cool() { sleep 25; }

echo "== single core latency (P-core 0, 5 repeats)"
for m in pmsm_heads_small_s0 pmsm_heads_medium_s0 pmsm_heads_large_s0 pmsm_heads_medium_s0_table64x64x128 pmsm_heads_large_s0_table64x64x128; do
    $L $R/$m.sldm $R/pmsm_heads_stream.bin 5 0 1 0
done
for m in pmsm_raw_w64_s0 pmsm_raw_w512_s0; do $L $R/$m.sldm $R/pmsm_stream.bin 5 0 1 0; done
for m in sarcos_full_angle_affine_small_s0 sarcos_full_angle_affine_medium_s0 sarcos_full_angle_affine_large_s0 sarcos_raw_w512_s0; do
    $L $R/$m.sldm $R/sarcos_stream.bin 5 0 1 1
done
$L $R/hyq_full_small_s0.sldm $R/hyq_stream_raw.bin 5 0 1 1
cool

echo "== F3: HyQ FULL small, step + observe, 12 threads: sessions vs fleets of 1024 and 4096, interleaved x3"
for k in 1 2 3; do
    $L $R/hyq_full_small_s0.sldm $R/hyq_stream_raw.bin 1 -1 12 1; cool
    $L $R/hyq_full_small_s0.sldm $R/hyq_stream_raw.bin 1 -1 12 1 1024; cool
    $L $R/hyq_full_small_s0.sldm $R/hyq_stream_raw.bin 1 -1 12 1 4096; cool
done
echo "== F3: PMSM RAW w64, step only, 12 threads: sessions vs fleet of 4096, interleaved x3"
for k in 1 2 3; do
    $L $R/pmsm_raw_w64_s0.sldm $R/pmsm_stream.bin 1 -1 12 0; cool
    $L $R/pmsm_raw_w64_s0.sldm $R/pmsm_stream.bin 1 -1 12 0 4096; cool
done

echo "== T1: PMSM tables (default grid 64x64x128 and the pre-registered 32x32x64), step only, 12 threads, x3"
for k in 1 2 3; do
    $L $R/pmsm_heads_medium_s0_table64x64x128.sldm $R/pmsm_heads_stream.bin 1 -1 12 0 4096; cool
    $L $R/pmsm_heads_medium_s0_table64x64x128.sldm $R/pmsm_heads_stream.bin 1 -1 12 0 65536; cool
    $L $R/pmsm_heads_medium_s0_table.sldm $R/pmsm_heads_stream.bin 1 -1 12 0 4096; cool
done

echo "== networks: fleets of 4096 vs sessions, 12 threads"
for m in pmsm_heads_small_s0 pmsm_heads_medium_s0 pmsm_heads_large_s0; do
    $L $R/$m.sldm $R/pmsm_heads_stream.bin 1 -1 12 0; cool
    $L $R/$m.sldm $R/pmsm_heads_stream.bin 1 -1 12 0 4096; cool
done
$L $R/pmsm_raw_w512_s0.sldm $R/pmsm_stream.bin 1 -1 12 0; cool
$L $R/pmsm_raw_w512_s0.sldm $R/pmsm_stream.bin 1 -1 12 0 1024; cool
for m in sarcos_full_angle_affine_small_s0 sarcos_full_angle_affine_medium_s0; do
    $L $R/$m.sldm $R/sarcos_stream.bin 1 -1 12 1; cool
    $L $R/$m.sldm $R/sarcos_stream.bin 1 -1 12 1 1024; cool
done
$L $R/sarcos_raw_w512_s0.sldm $R/sarcos_stream.bin 1 -1 12 0; cool
$L $R/sarcos_raw_w512_s0.sldm $R/sarcos_stream.bin 1 -1 12 0 1024; cool
echo "== HyQ step only (planning), 12 threads"
$L $R/hyq_full_small_s0.sldm $R/hyq_stream_raw.bin 1 -1 12 0; cool
$L $R/hyq_full_small_s0.sldm $R/hyq_stream_raw.bin 1 -1 12 0 1024
echo "== done"
