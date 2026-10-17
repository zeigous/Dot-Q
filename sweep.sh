#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

JOBS=8
SEEDS=(1 2 3)

U=sw::universal
# cfloat<nbits, es, storage, subnormals, supernormals=false, saturating=true>
cf() { echo "$U::cfloat<$1,$2,$3,$4,false,true>"; }
QCAP=31   # quire capacity

# Queue order within each seed = slowest first ./sweep.sh 2>&1 | tee smoke.log
FWD_NAMES=( posit16_2 posit16_1 posit8_2 posit8_1 fp16 bf16
             e4m3 e5m2
           )
           # lattitude: fp16_ftz bf16_ftz / pi: e4m3_ftz e5m2_ftz  posit8_0
declare -A FWD=(
  [posit16_2]="$U::posit<16,2>"
  [posit16_1]="$U::posit<16,1>"
  [posit8_2]="$U::posit<8,2>"
  [posit8_1]="$U::posit<8,1>"
  [posit8_0]="$U::posit<8,0>"
  [fp16]="$(cf 16 5 uint16_t true)"
  [bf16]="$(cf 16 8 uint16_t true)"
  [e4m3]="$(cf 8 4 uint8_t true)"
  [e5m2]="$(cf 8 5 uint8_t true)"
  [fp16_ftz]="$(cf 16 5 uint16_t false)"
  [bf16_ftz]="$(cf 16 8 uint16_t false)"
  [e4m3_ftz]="$(cf 8 4 uint8_t false)"
  [e5m2_ftz]="$(cf 8 5 uint8_t false)"
)

MUL_NAMES=(quire norm)    # accumulation method
BWD_NAMES=(same double)   # same = REAL_BWD is REAL_FWD

DS_NAMES=(svhn cifar10)
declare -A DS_TRAIN=( [cifar10]="datasets/CIFAR-10/cifar10_train.csv" [svhn]="datasets/SVHN/svhn_train.csv" )
declare -A DS_TEST=(  [cifar10]="datasets/CIFAR-10/cifar10_test.csv"  [svhn]="datasets/SVHN/svhn_test.csv" )

# SMOKE=1 ./sweep.sh -> one seed, one dataset, 3 types
if [[ "${SMOKE:-0}" == 1 ]]; then
  SEEDS=(1)
  DS_NAMES=(svhn)
  FWD_NAMES=(posit16_2 fp16 fp32)
  JOBS=3
fi

# todo, run one in true float and true double

COMMIT=$(git rev-parse --short HEAD)
git diff --quiet || COMMIT="${COMMIT}-dirty"
mkdir -p build out logs

# args: id fwd bwd accf accb mulf mulb train test seed
run_job() {
  local id=$1 fwd=$2 bwd=$3 accf=$4 accb=$5 mulf=$6 mulb=$7 train=$8 test=$9 seed=${10}
  local dir="build/$id" log="logs/$id.log"

  # Skip finished runs: 1 "Loss,Accuracy" header + 9 result rows = 10 non-comment lines
  if [[ -s "out/$id.csv" ]] && [[ $(grep -vc '^#' "out/$id.csv") -ge 10 ]]; then
    echo "skip $id (done)"
    return 0
  fi

  echo "building $id"
  rm -rf "$dir"
  {
    cmake -S . -B "$dir" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-DRANDOM_SEED=$seed -DREAL_FWD='$fwd' -DREAL_BWD='$bwd' -DACC_FWD='$accf' -DACC_BWD='$accb' -DMUL_FWD='$mulf' -DMUL_BWD='$mulb' -DDATASET_TRAIN='\"$train\"' -DDATASET_TEST='\"$test\"'" &&
    cmake --build "$dir"
  } > "$log" 2>&1 || { echo "BUILD FAILED $id (see $log)" >&2; rm -rf "$dir"; return 1; }

  {
    echo "# id=$id fwd=$fwd bwd=$bwd accf=$accf accb=$accb mulf=$mulf mulb=$mulb seed=$seed commit=$COMMIT"
    "$dir/main"
  } > "out/$id.csv" || { echo "RUN FAILED $id" >&2; rm -rf "$dir"; return 1; }

  rm -rf "$dir"
  echo "done $id"
}
export -f run_job
export COMMIT

# Seed is the OUTERMOST loop: all of seed 1, then all of seed 2, then seed 3
for seed in "${SEEDS[@]}"; do
for f in "${FWD_NAMES[@]}"; do
for m in "${MUL_NAMES[@]}"; do
for b in "${BWD_NAMES[@]}"; do
for ds in "${DS_NAMES[@]}"; do
  fwd="${FWD[$f]}"
  if [[ $m == quire ]]; then accf="$U::quire<$fwd,$QCAP>"; mulf=QUIRE_MUL
  else                       accf="$fwd";                  mulf=NORM_MUL; fi
  if [[ $b == double ]]; then bwd=double; accb=double; mulb=NORM_MUL
  else                        bwd="$fwd"; accb="$accf";  mulb="$mulf"; fi
  printf '%s\0' "${ds}_${f}_${m}_b${b}_s${seed}" "$fwd" "$bwd" "$accf" "$accb" "$mulf" "$mulb" \
                "${DS_TRAIN[$ds]}" "${DS_TEST[$ds]}" "$seed"
done
done
done
done
done | xargs -0 -n 10 -P "$JOBS" bash -c 'run_job "$@"' _