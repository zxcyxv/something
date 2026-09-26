"""Build offline interactive viewers and whole-board figures from captures."""

import argparse
import base64
import gzip
import json
from pathlib import Path

import numpy as np


def encode_payload(data, metadata):
    descriptors = {}
    parts = []
    cursor = 0

    def add(name, array, encoding='raw', scale=1.0):
        nonlocal cursor
        pad = (-cursor) % 4
        if pad:
            parts.append(b'\0' * pad)
            cursor += pad
        array = np.ascontiguousarray(array)
        content = array.tobytes()
        descriptors[name] = {'offset': cursor, 'count': int(array.size),
                             'shape': list(array.shape), 'dtype': array.dtype.str,
                             'encoding': encoding, 'scale': float(scale)}
        parts.append(content)
        cursor += len(content)

    errors = {}
    for j, name in enumerate(metadata['field_names']):
        values = data['matrices'][j]
        scale = max(float(np.abs(values).max()), 1e-8) / 32760
        quantized = np.rint(values / scale).astype('<i2')
        errors[name] = float(np.max(np.abs(quantized.astype(np.float32) * scale - values)))
        delta = quantized.copy()
        delta[1:] = np.subtract(quantized[1:], quantized[:-1], dtype=np.int16)
        add(name, delta, 'temporal_delta_i16', scale)
        if name == 'w':
            add('wInitial', np.rint(data['w_initial'] / scale).astype('<i2'), scale=scale)
    for target, source in [('signals', 'digit_signals'), ('qLogits', 'q_logits'),
                           ('denominators', 'denominators'), ('logits', 'logits'),
                           ('predictions', 'predictions'), ('beforePredictions', 'predictions_before')]:
        add(target, data[source])
    packed = gzip.compress(b''.join(parts), compresslevel=6, mtime=0)
    metadata['arrays'] = descriptors
    metadata['display_quantization_max_error'] = errors
    metadata['payload_uncompressed_bytes'] = cursor
    metadata['payload_compressed_bytes'] = len(packed)
    return base64.b64encode(packed).decode('ascii')


def draw_board(ax, pred, givens, gold, title, selected=None):
    import matplotlib.patches as patches
    ax.set_xlim(-0.45, 9)
    ax.set_ylim(9, -0.45)
    ax.set_aspect('equal')
    ax.axis('off')
    for c in range(81):
        r, col = divmod(c, 9)
        wrong = pred[c] != gold[c]
        color = '#ffe3e6' if wrong else ('#e7edf5' if givens[c] else '#ffffff')
        ax.add_patch(patches.Rectangle((col, r), 1, 1, facecolor=color, edgecolor='none'))
        ax.text(col + .5, r + .52, str(int(pred[c])) if pred[c] > 0 else '.',
                ha='center', va='center', fontsize=10, fontweight='bold' if givens[c] else 'normal',
                color='#b42339' if wrong else '#243249')
    for k in range(10):
        width = 1.6 if k % 3 == 0 else .35
        ax.plot([0, 9], [k, k], color='#64748b', lw=width)
        ax.plot([k, k], [0, 9], color='#64748b', lw=width)
    if selected is not None:
        r, c = divmod(selected, 9)
        ax.add_patch(patches.Rectangle((c + .035, r + .035), .93, .93, fill=False,
                                      edgecolor='#7c3aed', linewidth=2))
    ax.set_title(title, fontsize=10, loc='left', pad=9)


def make_figure(meta, data, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    offsets = np.asarray(meta['blocks']) - meta['event']['stable_complete_block']
    selected_offsets = [-128, -16, -1, 0, 16]
    fig = plt.figure(figsize=(16, 8.4), facecolor='#f7f9fc')
    grid = fig.add_gridspec(3, 5, height_ratios=[1.65, .9, .85], hspace=.57, wspace=.28)
    for j, offset in enumerate(selected_offsets):
        frame = int(np.abs(offsets - offset).argmin())
        metric = meta['metrics'][frame]
        draw_board(fig.add_subplot(grid[0, j]), data['predictions'][frame], meta['givens'], meta['gold'],
                   f"T{offsets[frame]:+d}   wrong {metric['wrong_cells']} / conflicts {metric['conflict_pairs']}",
                   meta['event']['cell'])
    ax = fig.add_subplot(grid[1, :])
    for key, label, color in [('wrong_cells', 'Wrong cells (gold reference)', '#e04f65'),
                              ('conflict_pairs', 'Duplicate peer pairs', '#d98b12'),
                              ('changed_cells', 'Changed predictions per block', '#3182ce')]:
        ax.plot(offsets, [m[key] for m in meta['metrics']], label=label, color=color, lw=1.5)
    ax.axvline(0, color='#7c3aed', ls='--', lw=1)
    ax.set_xlim(offsets[0], offsets[-1])
    ax.set_ylabel('Count')
    ax.set_xlabel('Blocks relative to stable puzzle completion T')
    ax.legend(loc='upper right', frameon=False, ncol=3, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.15)
    ax = fig.add_subplot(grid[2, :])
    incorrect = data['predictions'] != np.asarray(meta['gold'])[None]
    ax.imshow(incorrect.T, aspect='auto', interpolation='nearest', cmap='Reds', vmin=0, vmax=1,
              extent=[offsets[0] - .5, offsets[-1] + .5, 81, 0])
    ax.axvline(0, color='#7c3aed', ls='--', lw=1)
    ax.set_yticks(np.arange(4.5, 81, 9), [f'row {r}' for r in range(1, 10)])
    for k in range(0, 82, 9):
        ax.axhline(k, color='#cbd5e1', lw=.4)
    ax.set_xlabel('All 81 cells; red = incorrect, white = correct')
    fig.suptitle(f"Puzzle {meta['puzzle']}  |  v1.1 160k FP32  |  T = block {meta['event']['stable_complete_block']}",
                 x=.07, ha='left', fontsize=17, fontweight='bold')
    fig.text(.07, .02, 'Gray cells: original clues. Pink cells: wrong predictions. Purple outline: previously studied target. '
             'All predictions are decoded after the indicated block.', fontsize=9, color='#475569')
    fig.savefig(path, dpi=160, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def make_attention_figures(meta, data, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    offsets = np.asarray(meta['blocks']) - meta['event']['stable_complete_block']
    frames = [int(np.abs(offsets - k).argmin()) for k in [-16, -1, 0, 16]]
    psi, beta, agree, w = data['matrices']
    gain = np.asarray(meta['head_parameters']['gain'], dtype=np.float32)[None, :, None, None]
    lam = np.asarray(meta['head_parameters']['lambda'], dtype=np.float32)[None, :, None, None]
    fields = [('Read kernel a_psi', psi), ('Write kernel a_beta', beta),
              ('Write target G = gain * a_beta * agree', gain * beta * agree),
              ('Memory W (after write)', w), ('Read coefficient J', (1-lam)*psi + lam*w)]
    path = root / f"puzzle_{meta['puzzle']}_attention.pdf"
    with PdfPages(path) as pdf:
        # First page compares kernels and memory for the previously studied head.
        # The following pages expose all heads rather than averaging signed values.
        for chosen_head in [2, None]:
            groups = [fields] if chosen_head is not None else [[field] for field in fields]
            for group in groups:
                rows = len(group) if chosen_head is not None else 8
                fig, axes = plt.subplots(rows, 4, figsize=(13, rows * 2.55 + 1), squeeze=False)
                fig.subplots_adjust(left=.10, right=.89, bottom=.04, top=.94, hspace=.30, wspace=.16)
                for row in range(rows):
                    label, values = group[row] if chosen_head is not None else group[0]
                    head = chosen_head if chosen_head is not None else row
                    # One symmetric scale for every captured block and every head of a field.
                    lim = max(float(np.abs(values).max()), 1e-8)
                    for col, frame in enumerate(frames):
                        ax = axes[row, col]
                        im = ax.imshow(values[frame, head], cmap='RdBu_r', vmin=-lim, vmax=lim,
                                       interpolation='nearest', origin='upper', rasterized=True)
                        ax.set_xticks([0, 27, 54, 80], ['1', '28', '55', '81'], fontsize=7)
                        ax.set_yticks([0, 27, 54, 80], ['1', '28', '55', '81'], fontsize=7)
                        if row == 0:
                            ax.set_title(f'T{offsets[frame]:+d}', fontsize=11)
                        if col == 0:
                            ax.set_ylabel(label if chosen_head is not None else f'Head {head+1}', fontsize=9)
                    fig.colorbar(im, ax=axes[row, :].tolist(), fraction=.018, pad=.025, shrink=.9)
                title = f"Puzzle {meta['puzzle']} | " + ('Head 3: full-board connections' if chosen_head is not None else group[0][0] + ': all 8 heads')
                fig.suptitle(title, fontsize=15, y=.982)
                fig.text(.10, .009, 'Rows: destination cell. Columns: source cell. Cell order: r1c1 ... r9c9. Blue < 0 < red. Fixed scale across time / heads.', fontsize=8)
                pdf.savefig(fig, dpi=125)
                if chosen_head is not None:
                    fig.savefig(root / f"puzzle_{meta['puzzle']}_attention.png", dpi=125)
                plt.close(fig)
    print('ATTENTION_FIGURES', meta['puzzle'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/puzzle_overview_v11'))
    parser.add_argument('--puzzles', type=int, nargs='+', default=[58, 209, 230])
    args = parser.parse_args()
    template = (Path(__file__).parent / 'puzzle_overview_template.html').read_text()
    summaries = []
    for puzzle in args.puzzles:
        directory = args.root / f'puzzle_{puzzle}'
        meta = json.loads((directory / 'metadata.json').read_text())
        with np.load(directory / 'states_fp32.npz') as archive:
            data = {key: archive[key] for key in archive.files}
        print('PACKING', puzzle, flush=True)
        payload = encode_payload(data, meta)
        html = template.replace('__METADATA_JSON__', json.dumps(meta, ensure_ascii=False, separators=(',', ':')))
        html = html.replace('__PAYLOAD_BASE64__', payload)
        path = args.root / f'puzzle_{puzzle}.html'
        path.write_text(html)
        make_figure(meta, data, args.root / f'puzzle_{puzzle}_overview.png')
        make_attention_figures(meta, data, args.root)
        summaries.append({'puzzle': puzzle, 'html': path.name, 'bytes': path.stat().st_size,
                          'compressed_bytes': meta['payload_compressed_bytes'],
                          'display_error': meta['display_quantization_max_error']})
        print('BUILT', json.dumps(summaries[-1]), flush=True)
    (args.root / 'viewer_build.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + '\n')
    cards = ''.join(f'<a href="puzzle_{p}.html"><h2>퍼즐 {p}</h2><img src="puzzle_{p}_overview.png" alt="퍼즐 {p} 전체 풀이 흐름"><p>전후 128블록 · 81개 셀 · 8개 헤드 · 클릭하여 열기</p></a>' for p in args.puzzles)
    (args.root / 'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>Sudoku 풀이 동역학</title>'
        '<style>body{font:16px system-ui;background:#f4f7fb;color:#182840;max-width:1200px;margin:40px auto;padding:24px}a{display:block;background:white;border:1px solid #dde5ef;border-radius:16px;padding:24px;margin:20px 0;text-decoration:none;color:inherit}img{width:100%}p{color:#60718a}</style>'
        '<h1>Sudoku · 전체 풀이 동역학</h1><p>v1.1 160k FP32. 각 파일은 데이터를 포함하므로 오프라인으로 열 수 있습니다.</p>' + cards + '</html>')


if __name__ == '__main__':
    main()
