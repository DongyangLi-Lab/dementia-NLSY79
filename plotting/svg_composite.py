"""Compose native Matplotlib SVGs without reinterpreting PDF transparency masks."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET

NS='http://www.w3.org/2000/svg';XLINK='http://www.w3.org/1999/xlink'
ET.register_namespace('',NS);ET.register_namespace('xlink',XLINK)

def compose(top,bottom,destination):
 width=190*72/25.4;height=226*72/25.4;offset=114*72/25.4
 root=ET.Element(f'{{{NS}}}svg',{'width':f'{width}pt','height':f'{height}pt','viewBox':f'0 0 {width} {height}','version':'1.1'})
 for file,prefix,dy in [(top,'density_',0),(bottom,'selected_',offset)]:
  source=ET.parse(file).getroot();ids={e.attrib['id']:prefix+e.attrib['id'] for e in source.iter() if 'id' in e.attrib}
  for e in source.iter():
   if 'id' in e.attrib:e.set('id',ids[e.attrib['id']])
   for key,value in list(e.attrib.items()):
    if key.endswith('href') and value.startswith('#') and value[1:] in ids:value='#'+ids[value[1:]]
    value=re.sub(r'url\(#([^\)]+)\)',lambda m:'url(#'+ids.get(m[1],m[1])+')',value)
    e.set(key,value)
  container=ET.SubElement(root,f'{{{NS}}}g',{'id':prefix+'panel','transform':f'translate(0,{dy})'})
  for child in list(source):container.append(child)
 text=ET.SubElement(root,f'{{{NS}}}text',{'x':str(4*72/25.4),'y':str(115.5*72/25.4),'font-family':'Arial','font-size':'11','font-weight':'bold','fill':'#25252B'})
 text.text='G'
 Path(destination).write_bytes(ET.tostring(root,encoding='utf-8',xml_declaration=True))

if __name__=='__main__':
 o=Path(__file__).resolve().parents[1]
 compose(o/'Fig3_density_component.svg',o/'selected_component'/'education_navigation_violin_box_overlay.svg',o/'Fig3_functional_landscape.svg')
