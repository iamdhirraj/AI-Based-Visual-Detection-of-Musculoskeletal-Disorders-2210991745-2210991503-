from __future__ import annotations

from pathlib import Path

OUTPUT = Path(__file__).resolve().parent / "assets" / "mura-corrected-report.pdf"

WIDTH = 595.28
HEIGHT = 841.89


def esc(text: str) -> str:
    return text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')


def text_ops(lines: list[str], x: float, y: float, size: int = 12, leading: int | None = None, font: str = 'F1') -> str:
    if leading is None:
        leading = int(size * 1.35)
    commands = [f'BT /{font} {size} Tf {leading} TL 1 0 0 1 {x:.2f} {y:.2f} Tm']
    if lines:
        commands.append(f'({esc(lines[0])}) Tj')
        for line in lines[1:]:
            commands.append(f'T* ({esc(line)}) Tj')
    commands.append('ET')
    return '\n'.join(commands)


def make_chart_bar(x: float, y: float, width: float, height: float, color: tuple[float, float, float]) -> str:
    r, g, b = color
    return f'{r:.3f} {g:.3f} {b:.3f} rg {x:.2f} {y:.2f} {width:.2f} {height:.2f} re f'


def build_content() -> str:
    lines = []
    lines.append('0.12 0.18 0.24 rg 0 0 0 RG')
    lines.append(make_chart_bar(0, 770, WIDTH, 72, (0.08, 0.31, 0.34)))
    lines.append(text_ops(['AI-Based Visual Detection of Musculoskeletal Disorders'], 44, 800, size=24, leading=28, font='F1'))
    lines.append(text_ops(['Personalized exercise recommendations powered by deep learning.'], 44, 778, size=11, leading=14, font='F2'))

    lines.append(text_ops(['Key facts'], 44, 724, size=16, leading=18, font='F1'))
    lines.append(text_ops([
        'Dataset: MURA musculoskeletal radiographs',
        'Total studies: 14,863',
        'Normal studies: 9,045',
        'Abnormal studies: 5,818',
        'Input size used in the notebook: 256 x 256',
    ], 44, 704, size=11, leading=16, font='F2'))

    lines.append(text_ops(['Model coverage'], 44, 606, size=16, leading=18, font='F1'))
    lines.append(text_ops([
        'CNN variants: 4',
        'VGG variants: 2',
        'ResNet variants: 2',
        'DenseNet variants: 5',
        'Total model sections: 13',
    ], 44, 586, size=11, leading=16, font='F2'))

    lines.append(text_ops(['Notebook flow'], 44, 498, size=16, leading=18, font='F1'))
    lines.append(text_ops([
        '1. Load and label data from the MURA directory structure.',
        '2. Resize and normalize images, with optional augmentation.',
        '3. Train multiple CNN, VGG, ResNet, and DenseNet architectures.',
        '4. Evaluate with AUC, accuracy, and Cohen\'s kappa.',
    ], 44, 478, size=11, leading=16, font='F2'))

    lines.append(text_ops(['Label balance chart'], 332, 724, size=16, leading=18, font='F1'))
    lines.append('0.08 0.46 0.43 rg')
    lines.append('332 668 210 24 re f')
    lines.append('0.68 0.49 0.24 rg')
    lines.append('332 636 135 24 re f')
    lines.append('0.9 0.9 0.9 rg')
    lines.append('332 604 135 24 re f')
    lines.append(text_ops(['Normal', '9,045'], 552, 676, size=10, leading=13, font='F2'))
    lines.append(text_ops(['Abnormal', '5,818'], 552, 644, size=10, leading=13, font='F2'))
    lines.append(text_ops(['The bar lengths are scaled to the notebook-provided counts.'], 332, 578, size=10, leading=13, font='F2'))

    lines.append(text_ops(['Correction note'], 44, 410, size=16, leading=18, font='F1'))
    lines.append(text_ops([
        'This PDF replaces the incorrect attachment with a clean, notebook-derived report.',
        'It is intentionally concise so it can be used as a dependable reference alongside the website.',
    ], 44, 390, size=11, leading=16, font='F2'))

    lines.append('0.12 0.18 0.24 rg')
    lines.append('0 38 595.28 40 re f')
    lines.append(text_ops(['AI-Powered Musculoskeletal Disorder Detection System'], 44, 54, size=10, leading=13, font='F2'))
    lines.append(text_ops(['Detection & Rehabilitation'], 474, 54, size=10, leading=13, font='F2'))
    return '\n'.join(lines)


def build_pdf() -> bytes:
    content = build_content().encode('utf-8')

    objects: list[bytes] = []
    objects.append(b'<< /Type /Catalog /Pages 2 0 R >>')
    objects.append(b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>')
    objects.append(
        f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {WIDTH:.2f} {HEIGHT:.2f}] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>'.encode(
            'utf-8'
        )
    )
    objects.append(b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>')
    objects.append(b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>')
    objects.append(f'<< /Length {len(content)} >>\nstream\n'.encode('utf-8') + content + b'\nendstream')

    pdf = bytearray()
    pdf.extend(b'%PDF-1.4\n')
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f'{index} 0 obj\n'.encode('utf-8'))
        pdf.extend(obj)
        pdf.extend(b'\nendobj\n')

    xref_offset = len(pdf)
    pdf.extend(f'xref\n0 {len(objects) + 1}\n'.encode('utf-8'))
    pdf.extend(b'0000000000 65535 f \n')
    for offset in offsets[1:]:
        pdf.extend(f'{offset:010d} 00000 n \n'.encode('utf-8'))
    pdf.extend(
        f'trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n'.encode('utf-8')
    )
    return bytes(pdf)


if __name__ == '__main__':
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(build_pdf())
    print(f'Wrote {OUTPUT}')
