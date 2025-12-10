# Eligibility Traces in R-STDP: How They Work

## Overview

In Lava-nc, eligibility traces are stored in the variable `tag_1` of the `LearningDense` process. They are updated according to a differential equation (`dt`) defined in the learning rule.

## The R-STDP Learning Rule

The `RewardModulatedSTDP` learning rule from `src/lava/proc/learning_rules/r_stdp_learning_rule.py` defines two update formulas:

### 1. Eligibility Trace Update (`dt`)

$$
\Delta t = \eta \cdot A^- \cdot x_0 \cdot y_1 + \eta \cdot A^+ \cdot y_0 \cdot x_1 - u \cdot t \cdot \tau_{\text{elig}}
$$

Where:
- $t$ = eligibility trace (stored in `tag_1`)
- $\eta$ = learning rate
- $A^+, A^-$ = potentiation and depression scaling factors
- $x_0$ = 1 if pre-synaptic spike occurred in this epoch, 0 otherwise
- $y_0$ = 1 if post-synaptic spike occurred in this epoch, 0 otherwise
- $x_1$ = pre-synaptic trace (exponentially decaying after pre spike)
- $y_1$ = post-synaptic trace (exponentially decaying after post spike)
- $u$ = "epoch clock" dependency (1 every epoch with `decimate_exponent=0`)
- $\tau_{\text{elig}}$ = eligibility decay factor (NOT a time constant!)

### 2. Weight Update (`dw`)

$$
\Delta w = u \cdot t \cdot y_2
$$

Where:
- $w$ = synaptic weight
- $t$ = eligibility trace
- $y_2$ = reward signal (third factor)

## Critical Issue: Eligibility Tau Naming

**The parameter `eligibility_trace_decay_tau` is MISLEADING!**

In the formula, `τ_elig` is used as a direct multiplication factor in the decay term:

$$
t_{\text{new}} = t_{\text{old}} + \Delta t = t_{\text{old}} - u \cdot t_{\text{old}} \cdot \tau_{\text{elig}} + \text{(spike terms)}
$$

For stable decay (ignoring spike terms):

$$
t_{\text{new}} = t_{\text{old}} \cdot (1 - \tau_{\text{elig}})
$$

This means:
- If `eligibility_tau = 20.0`: $t_{\text{new}} = t_{\text{old}} \cdot (1 - 20) = -19 \cdot t_{\text{old}}$ → **WILDLY UNSTABLE!**
- If `eligibility_tau = 0.05`: $t_{\text{new}} = t_{\text{old}} \cdot 0.95$ → stable exponential decay with τ ≈ 20 steps

### Correct Values

For an effective time constant of $\tau$ steps:

$$
\tau_{\text{elig}} = \frac{1}{\tau}
$$

Examples:
| Desired time constant | `eligibility_tau` value |
|----------------------|------------------------|
| 10 steps | 0.10 |
| 20 steps | 0.05 |
| 50 steps | 0.02 |
| 100 steps | 0.01 |

## Current Code Status

### Backend (`backend.py`)

All traces are now read directly from the model:

```python
# Read processes for monitoring
x1_reader = Read(buffer=num_steps, interval=1, offset=0)
y1_reader = Read(buffer=num_steps, interval=1, offset=0)
tag_reader = Read(buffer=num_steps, interval=1, offset=0)

x1_reader.connect_var(plast_conn.x1)   # Pre-synaptic trace
y1_reader.connect_var(plast_conn.y1)   # Post-synaptic trace (REAL, not approximation)
tag_reader.connect_var(plast_conn.tag_1)  # Eligibility trace
```

Post traces are extracted from real y1 data:

```python
y1_data = raw_y1.reshape(num_steps, 2, 1)
post_trace_A = y1_data[:, 0, 0]
post_trace_B = y1_data[:, 1, 0]
```

### Server (`server.py`)

Eligibility traces are sent in the frame:

```python
"eligibility": [
    rstdp["eligibility_A"][i],
    rstdp["eligibility_B"][i],
]
```

### Frontend (`app.js`)

Eligibility traces are received and drawn:

```javascript
rstdp.eligibilityA.push(frame.eligibility ? frame.eligibility[0] : 0);
rstdp.eligibilityB.push(frame.eligibility ? frame.eligibility[1] : 0);
// ...
drawDualTrace(rstdpCtx.eligibility, ..., rstdp.eligibilityA, rstdp.eligibilityB, ..., true);
```

The `allowNegative = true` parameter allows correct display of negative eligibility values.

## Why the Graphs Look Strange

With `eligibility_tau = 20.0`, the decay term `-u * t * 20` causes:

1. **Sign oscillation**: Each update multiplies the trace by $(1 - 20) = -19$
2. **Exponential explosion**: Values grow exponentially while flipping sign
3. **Chaotic behavior**: The trace oscillates wildly between large positive and negative values

## Recommended Fix

Change the default `eligibility_tau` in `backend.py`:

```python
eligibility_tau: float = 0.05,  # Effective time constant ≈ 20 steps
```

Or use even smaller values for longer memory:

```python
eligibility_tau: float = 0.02,  # Effective time constant ≈ 50 steps
```

## Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Reading `tag_1` (eligibility) | ✅ Correct | Uses `Read` process on `plast_conn.tag_1` |
| Reading `x1` (pre-trace) | ✅ Correct | Uses `Read` process on `plast_conn.x1` |
| Reading `y1` (post-trace) | ✅ Correct | Uses `Read` process on `plast_conn.y1` |
| Sending to frontend | ✅ Correct | All traces sent in JSON frame |
| Display in frontend | ✅ Correct | Uses `allowNegative=true` for eligibility |
| **Parameter value** | ✅ Fixed | `eligibility_tau=0.05` (effective τ ≈ 20 steps) |

All traces now come directly from the Lava model, ensuring synchronization between displayed values and actual learning dynamics.

