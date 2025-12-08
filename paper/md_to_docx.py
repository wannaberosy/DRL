import os
import zipfile
from datetime import datetime
from typing import List, Optional, Tuple


CONTENT_TYPES = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
'''

RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
  <Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="word/styles.xml"/>
</Relationships>
'''

APP_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Trae IDE</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
  <Company></Company>
  <LinksUpToDate>false</LinksUpToDate>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>16.0000</AppVersion>
  <Pages>1</Pages>
  <Words>0</Words>
  <Characters>0</Characters>
  <Lines>1</Lines>
</Properties>
'''

def core_xml(title="report", creator="DRL Project"):
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{title}</dc:title>
  <dc:creator>{creator}</dc:creator>
  <cp:lastModifiedBy>{creator}</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
  <cp:revision>1</cp:revision>
</cp:coreProperties>
'''

STYLES_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:outlineLvl w:val="0"/>
    </w:pPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:outlineLvl w:val="1"/>
    </w:pPr>
  </w:style>
</w:styles>
'''

def escape_text(s: str) -> str:
    return (s.replace('&', '&amp;')
             .replace('<', '&lt;')
             .replace('>', '&gt;'))

def make_run(text: str) -> str:
    parts = text.split('\n')
    runs = []
    for i, part in enumerate(parts):
        t = escape_text(part)
        runs.append(f'<w:r><w:t xml:space="preserve">{t}</w:t></w:r>')
        if i < len(parts) - 1:
            runs.append('<w:r><w:br/></w:r>')
    return ''.join(runs)

def paragraph_xml(text: str, style: Optional[str]) -> str:
    run = make_run(text)
    if style:
        return f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>{run}</w:p>'
    return f'<w:p>{run}</w:p>'

def build_document_xml(paragraphs: List[Tuple[Optional[str], str]]) -> str:
    body = []
    for style, text in paragraphs:
        body.append(paragraph_xml(text, style))
    body_xml = ''.join(body)
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {body_xml}
    <w:sectPr/>
  </w:body>
</w:document>
'''

def md_to_paragraphs(md_text: str) -> List[Tuple[Optional[str], str]]:
    lines = md_text.splitlines()
    paragraphs = []
    buffer: list[str] = []
    for line in lines:
        if line.startswith('# '):
            if buffer:
                paragraphs.append((None, '\n'.join(buffer).strip()))
                buffer = []
            paragraphs.append(('Heading1', line[2:].strip()))
        elif line.startswith('## '):
            if buffer:
                paragraphs.append((None, '\n'.join(buffer).strip()))
                buffer = []
            paragraphs.append(('Heading2', line[3:].strip()))
        elif line.strip() == '':
            if buffer:
                paragraphs.append((None, '\n'.join(buffer).strip()))
                buffer = []
        else:
            buffer.append(line)
    if buffer:
        paragraphs.append((None, '\n'.join(buffer).strip()))
    return paragraphs

def write_docx(paragraphs: List[Tuple[Optional[str], str]], out_path: str):
    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', CONTENT_TYPES)
        z.writestr('_rels/.rels', RELS)
        z.writestr('docProps/app.xml', APP_XML)
        z.writestr('docProps/core.xml', core_xml())
        z.writestr('word/styles.xml', STYLES_XML)
        z.writestr('word/document.xml', build_document_xml(paragraphs))

def main():
    base = os.path.dirname(__file__)
    md_path = os.path.join(base, 'report.md')
    out_path = os.path.join(base, 'report.docx')
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()
    paragraphs = md_to_paragraphs(md_text)
    write_docx(paragraphs, out_path)
    print(f'生成完成: {out_path}')

if __name__ == '__main__':
    main()
