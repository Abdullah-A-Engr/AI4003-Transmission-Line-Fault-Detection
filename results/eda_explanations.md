# Task 2 — Exploratory plots

## Figure 1: Ia, Ib and Ic against row number

The three phase currents vary across the first 2,000 consecutive rows, and the shaded regions mark rows labelled as faults. The shading lets us compare the labelled fault intervals with changes in the current traces; the three phases also show related, rather than independent, current patterns.

## Figure 2: Ia against Ib, coloured by fault

No-fault observations form a relatively compact cluster near the centre, while fault observations cover much wider current ranges and form curved/loop-like patterns around and through that cluster. This overlap and curved structure means a single straight line in the `(Ia, Ib)` plane cannot cleanly separate all observations.

## Figure 3: Ia histograms for fault and no-fault rows

No-fault values of `Ia` are concentrated in a comparatively narrow region around zero, while fault values extend across a much wider range of positive and negative currents. There is still overlap around the central region, so `Ia` by itself cannot identify every row correctly.

## Can one straight line separate the two classes?

Not reliably in the `(Ia, Ib)` plane. The fault points form curved patterns and overlap the central region occupied by no-fault points, so one straight boundary cannot separate all cases. A fault can change several phase currents together, and adding voltage measurements, squared features, or a nonlinear kernel can represent more of that structure.
