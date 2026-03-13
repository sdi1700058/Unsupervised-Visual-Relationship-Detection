#!/usr/bin/env python3
"""
First-Order Logic (FOL) extraction utilities for FOSAE.

Converts FOSAE latent representations into human-readable FOL predicates.

Architecture recap:
  - Input: (batch, num_objs, num_features)  - object-centric state representation
  - Attention: (batch, U, A, num_objs)  - soft object selection per predicate unit/arg
  - Binary latent: (batch, U*P)  - truth values for U predicate-units x P predicates
  - Object bindings: argmax of attention -> which objects fill which argument slots

The FOL interpretation is:
  For each predicate unit u in {0..U-1}:
    bound_objects = [argmax_o attention[u, a, o] for a in {0..A-1}]
    For each predicate p in {0..P-1}:
      If latent[u, p] == 1:
        pred_p(obj_{bound_objects[0]}, obj_{bound_objects[1]}, ...) = TRUE
"""

import numpy as np
import json
import os


def extract_attention_bindings(attention, threshold=None):
    """Extract hard object bindings from soft attention weights.

    Args:
        attention: np.array of shape (batch, U, A, num_objs)
        threshold: if set, also return confidence scores. Objects with
                   attention below threshold are marked as uncertain.

    Returns:
        bindings: np.array of shape (batch, U, A)  - argmax object indices
        confidences: np.array of shape (batch, U, A)  - max attention values
    """
    bindings = np.argmax(attention, axis=-1)          # (batch, U, A)
    confidences = np.max(attention, axis=-1)           # (batch, U, A)
    return bindings, confidences


def extract_predicate_values(latent_codes, U, P):
    """Reshape flat latent codes into per-unit predicate truth values.

    Args:
        latent_codes: np.array of shape (batch, U*P)  - raw encoder output
        U: number of predicate units
        P: number of predicates per unit

    Returns:
        predicates: np.array of shape (batch, U, P)  - binary predicate values
    """
    rounded = np.round(latent_codes).astype(int)
    return rounded.reshape(-1, U, P)


def state_to_fol_atoms(bindings, predicates, confidences=None,
                       object_names=None, predicate_names=None,
                       confidence_threshold=0.6):
    """Convert a single state's bindings + predicates into FOL atom strings.

    Args:
        bindings: np.array of shape (U, A)  - object indices per  unit/arg
        predicates: np.array of shape (U, P)  - binary predicate values
        confidences: optional np.array of shape (U, A)  - attention confidences
        object_names: optional list of strings for object labels
        predicate_names: optional list of strings for predicate labels
        confidence_threshold: min attention confidence to include an atom

    Returns:
        atoms: list of dicts with keys:
            'unit', 'predicate', 'arguments', 'value', 'confidence', 'atom_str'
    """
    U, A = bindings.shape
    _, P = predicates.shape
    atoms = []

    for u in range(U):
        # Check confidence
        if confidences is not None:
            unit_conf = float(np.min(confidences[u]))
            if unit_conf < confidence_threshold:
                continue
        else:
            unit_conf = 1.0

        # Get bound object names
        obj_ids = bindings[u]  # (A,)
        if object_names is not None:
            arg_names = [object_names[oid] for oid in obj_ids]
        else:
            arg_names = [f"obj_{oid}" for oid in obj_ids]

        for p in range(P):
            value = bool(predicates[u, p])
            if predicate_names is not None and p < len(predicate_names):
                pname = predicate_names[p]
            else:
                pname = f"pred_{p}"

            args_str = ", ".join(arg_names)
            atom_str = f"{pname}({args_str})"
            if not value:
                atom_str = f"~{atom_str}"

            atoms.append({
                'unit': int(u),
                'predicate_idx': int(p),
                'predicate_name': pname,
                'arguments': [int(x) for x in obj_ids],
                'argument_names': arg_names,
                'value': value,
                'confidence': float(unit_conf),
                'atom_str': atom_str,
            })

    return atoms


def extract_fol_from_model(ae, data, object_names=None, predicate_names=None,
                           confidence_threshold=0.6):
    """Extract FOL representation from a trained FOSAE for a batch of states.

    Args:
        ae: trained FirstOrderSAE instance (must be loaded)
        data: np.array of shape (batch, num_objs, num_features)
        object_names: optional list of object name strings shared across all
            states  OR  a list-of-lists of length == len(data) providing
            per-state names (used for mixed-scene datasets where each state
            has different objects, e.g. labeled COCO images).
        predicate_names: optional list of predicate name strings
        confidence_threshold: min attention confidence to include atoms

    Returns:
        results: list of dicts, one per state, each containing:
            'atoms': list of FOL atom dicts
            'active_atoms': list of only TRUE atoms
            'latent_code': the binary latent vector
            'bindings': object binding array
            'object_names': the name list used for this state
    """
    U = ae.parameters['U']
    P = ae.parameters['P']
    A = ae.parameters['A']

    # Detect whether object_names is per-state (list-of-lists) or shared (flat list)
    per_state_names = (
        object_names is not None
        and len(object_names) > 0
        and isinstance(object_names[0], (list, tuple))
    )

    # Get all representations
    attention = ae.encode_attention(data)  # (batch, U, A, num_objs)
    latent    = ae.encode(data)            # (batch, U*P)

    bindings, confidences = extract_attention_bindings(attention)
    predicates = extract_predicate_values(latent, U, P)

    results = []
    for i in range(len(data)):
        state_names = object_names[i] if per_state_names else object_names
        atoms = state_to_fol_atoms(
            bindings[i], predicates[i], confidences[i],
            object_names=state_names,
            predicate_names=predicate_names,
            confidence_threshold=confidence_threshold,
        )
        active_atoms = [a for a in atoms if a['value']]

        results.append({
            'state_idx': i,
            'atoms': atoms,
            'active_atoms': active_atoms,
            'active_atom_strs': [a['atom_str'] for a in active_atoms],
            'latent_code': latent[i].round().astype(int).tolist(),
            'bindings': bindings[i].tolist(),
            'confidences': confidences[i].tolist(),
            'object_names': list(state_names) if state_names is not None else None,
        })

    return results


def format_fol_state(result, show_negated=False, show_confidence=True):
    """Format a single state's FOL representation as a readable string.

    Args:
        result: dict from extract_fol_from_model
        show_negated: if True, also show negated predicates
        show_confidence: if True, show attention confidence

    Returns:
        formatted string
    """
    lines = []
    lines.append(f"State {result['state_idx']}:")

    atoms = result['atoms'] if show_negated else result['active_atoms']
    if not atoms:
        lines.append("  (no active predicates)")
        return "\n".join(lines)

    for atom in atoms:
        conf_str = f"  [conf={atom['confidence']:.3f}]" if show_confidence else ""
        lines.append(f"  {atom['atom_str']}{conf_str}")

    return "\n".join(lines)


def save_fol_to_json(results, filepath, parameters=None):
    """Save FOL extraction results to a JSON file.

    Args:
        results: list of dicts from extract_fol_from_model
        filepath: output path
        parameters: optional dict of model parameters to include
    """
    output = {
        'description': 'FOL predicate extraction from FOSAE latent space',
        'num_states': len(results),
    }
    if parameters:
        output['model_parameters'] = {
            'U': parameters.get('U'),
            'A': parameters.get('A'),
            'P': parameters.get('P'),
            'num_objs': parameters.get('num_objs'),
        }
    output['states'] = results

    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"FOL predicates saved to {filepath}")


def save_fol_to_text(results, filepath, show_negated=False):
    """Save FOL extraction as human-readable text file.

    Args:
        results: list of dicts from extract_fol_from_model
        filepath: output path
        show_negated: if True, include negated predicates
    """
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w') as f:
        for result in results:
            f.write(format_fol_state(result, show_negated=show_negated))
            f.write("\n\n")
    print(f"FOL text saved to {filepath}")


def analyze_predicate_usage(results, U, P):
    """Analyze how predicates are used across states.

    Returns a summary dict with statistics about each predicate.
    """
    n_states = len(results)
    # Count how often each predicate is active
    pred_counts = np.zeros((U, P), dtype=int)
    for result in results:
        code = np.array(result['latent_code']).reshape(U, P)
        pred_counts += code

    summary = {
        'total_states': n_states,
        'predicate_activation_rates': (pred_counts / n_states).tolist(),
        'always_on': [],
        'always_off': [],
        'variable': [],
    }
    for u in range(U):
        for p in range(P):
            rate = pred_counts[u, p] / n_states
            info = {'unit': u, 'predicate': p, 'activation_rate': round(rate, 4)}
            if rate > 0.99:
                summary['always_on'].append(info)
            elif rate < 0.01:
                summary['always_off'].append(info)
            else:
                summary['variable'].append(info)

    summary['num_always_on'] = len(summary['always_on'])
    summary['num_always_off'] = len(summary['always_off'])
    summary['num_variable'] = len(summary['variable'])

    return summary
