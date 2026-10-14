/* declarations/gpu_recorder_claim.h -- how the particle touched set is claimed from a kernel.
 *
 * The touched set records the local particles a walk reached, so only those are drifted: a
 * stamp array as long as the local particles, a compacted list, a cursor, and a generation that
 * invalidates every stamp in O(1).  It has two claimers -- a host phase and a device kernel --
 * and exactly ONE claim belongs to it, or the two drift apart.
 *
 * The claim lives here rather than beside its recorder because its body needs Kokkos atomics,
 * and mesh/neighbor_list.h is plain data read by host units no device compiler ever sees.
 * (The gravity tree's node dirty set follows the same idiom but is claimed and answered on the
 * host only, so its claim lives with its storage in gravity/gpu_gravity_tree.cc.)
 *
 * Include after Kokkos and allvars.h; every includer already has both.
 */
#ifndef GPU_RECORDER_CLAIM_H
#define GPU_RECORDER_CLAIM_H

#include "../mesh/neighbor_list.h"   /* struct GxTouchedSet, gx_touched_owner_t, the GX_WALK_ANOMALY_* codes -- plain data, no Kokkos */


/* ============================================================================
 * THE PARTICLE TOUCHED SET
 * ========================================================================== */

/* THE claim, and the only one -- the fused walk's recording visitor comes through here.
 *
 * `owner` is the phase the caller believes it is in, checked against the owner the host wrote
 * when it opened the epoch and carried by value in the view.  It is what stops one caller
 * appending into another's live generation: the touched set deliberately keeps ONE generation
 * across several passes of a fused call while resetting the cursor at each consume, so a second
 * claimer arriving mid-call would be indistinguishable from that call's own later pass.
 *
 * Overflow is structurally impossible rather than handled -- the stamp admits each owned slot
 * once per generation and the list is as long as there are owned slots -- so the full case is
 * reported, not recovered from.  There is no `unsafe` word in device memory for this recorder
 * for the same reason; the `anomaly` word both callers already carry is the whole channel. */
KOKKOS_INLINE_FUNCTION void
gx_touched_set_claim_in(const struct GxTouchedSet &ts, int j, int owner, int *anomaly)
{
    if(!ts.seen || !ts.list || !ts.counter) {
        if(anomaly) {Kokkos::atomic_store(anomaly, GX_WALK_ANOMALY_TOUCHED_SET_FULL);}
        return;
    }
    if(ts.owner != owner) {
        if(anomaly) {Kokkos::atomic_store(anomaly, GX_WALK_ANOMALY_RECORDER_OUT_OF_PHASE);}
        return;
    }
    if(j < 0 || j >= ts.capacity) {
        if(anomaly) {Kokkos::atomic_store(anomaly, GX_WALK_ANOMALY_TOUCHED_SET_FULL);}
        return;
    }
    /* Claim the slot for this generation.  The exchange returns what was there, so exactly one
     * work item sees a value other than the current generation and exactly one item appends --
     * several actives reaching the same particle is the ordinary case, not the exception. */
    if(Kokkos::atomic_exchange(&ts.seen[j], ts.gen) == ts.gen) {return;}
    const int slot = Kokkos::atomic_fetch_add(ts.counter, 1);
    if(slot < ts.capacity) {ts.list[slot] = j;}
    else if(anomaly)       {Kokkos::atomic_store(anomaly, GX_WALK_ANOMALY_TOUCHED_SET_FULL);}
}

/* Whether local particle j has mass (Mass > 0), as a fused walk's leaf tests it.
 *
 * With use_verdict, the answer is kept for the rest of the call in mass_verdict[j] (GX_MASS_VERDICT_NONE):
 * the first visit of a generation reads Mass from the particle record and publishes the answer with one
 * compare-and-swap, and later visits read the word.  The word is the whole answer, so a visit that sees
 * the current generation needs nothing else, and a visit that loses the swap has read the record itself.
 * use_verdict is set only for loops whose pair kernel cannot take a neighbour's mass to zero or below
 * (Spec::keeps_neighbour_mass_positive) and only on a device walk, where the verdicts live; otherwise Mass
 * is read at every visit.  Mass is read atomically because a pair kernel that deposits mass into its
 * neighbours may be adding to this one from another lane at this moment; such a deposit is positive, so
 * it cannot change the answer. */
KOKKOS_INLINE_FUNCTION bool
gx_candidate_has_mass(const struct GxTouchedSet &ts, int use_verdict, const struct particle_data *P, int j)
{
    const int kept = (use_verdict && ts.mass_verdict && j >= 0 && j < ts.capacity);
    const unsigned int v = kept ? Kokkos::atomic_load(&ts.mass_verdict[j]) : 0u;
    if(kept && v == ts.gen) {return true;}
    if(kept && v == (ts.gen | GX_MASS_VERDICT_NONE)) {return false;}
    const bool has_mass = (Kokkos::atomic_load(&P[j].Mass) > 0);
    if(kept) {Kokkos::atomic_compare_exchange(&ts.mass_verdict[j], v, has_mass ? ts.gen : (ts.gen | GX_MASS_VERDICT_NONE));}
    return has_mass;
}

#endif /* GPU_RECORDER_CLAIM_H */
