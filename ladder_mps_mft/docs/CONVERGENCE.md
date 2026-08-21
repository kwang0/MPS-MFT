# SCF convergence and recurrence handling

For all stored alpha, beta, and spin-resolved density fields, the code forms one field vector `x`. If `f(x)` is the newly measured mean-field map, it records

```text
r_abs = max |f(x)-x|
r_rel = ||f(x)-x||_2 / max(||f(x)||_2, ||x||_2, eps).
```

A field step passes when either the absolute or relative tolerance passes. The absolute test protects near-zero fields; the relative test prevents a large physical field from passing on absolute scale alone. A fixed point additionally requires the configured number of consecutive passing iterations, target density, change in canonical variational energy per physical site, Hamiltonian identity, and effective-eigenvalue consistency.

The measured-field history is searched for the smallest recurrence period from 1 through `max_period`. A candidate period p must reproduce each phase of the orbit over `period_repeats` links. Searching upward makes the reported value the fundamental detected period rather than an arbitrary multiple.

- Period 1 can be accepted only through all fixed-point gates.
- Period p>1 is stored with every cycle member, recurrence residuals, and `accepted=false`.
- With `cycle_action=stop`, the run terminates as `periodic_cycle`.
- With `cycle_action=continue`, an immutable cycle artifact is written, Anderson history is cleared, damping is reduced, and iteration continues. The archived cycle remains nonphysical unless a separate stability analysis justifies it.

The code never averages a cycle into a field and never marks it `completed=true`.

Other terminal statuses are `stagnated`, `diverging`, `nonfinite`, `time_limit`, and `maximum_iterations`. They all remain selectable only with the explicit incomplete-data option and are rendered as `hatched` by the selection layer.

Linear and Anderson mixing are available. Adaptive damping is reduced after residual growth and increased conservatively after improvement. Density targeting uses a safeguarded bracket/secant search with its own tolerance, iteration cap, and time deadline; it does not reuse the SCF field tolerance.
