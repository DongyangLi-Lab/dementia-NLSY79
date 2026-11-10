"""Equivalent SVG presentation attributes for readers with incomplete inline-CSS support."""
from pathlib import Path
import argparse
import xml.etree.ElementTree as ET

PROPERTIES={'fill','fill-opacity','fill-rule','stroke','stroke-opacity','stroke-width',
 'stroke-linecap','stroke-linejoin','stroke-miterlimit','stroke-dasharray','stroke-dashoffset','opacity'}
ET.register_namespace('','http://www.w3.org/2000/svg')
ET.register_namespace('xlink','http://www.w3.org/1999/xlink')

def normalize(path):
 path=Path(path);tree=ET.parse(path);count=0
 for element in tree.iter():
  style=element.get('style')
  if not style:continue
  remaining=[]
  for part in style.split(';'):
   if ':' not in part:continue
   name,value=part.split(':',1);name=name.strip();value=value.strip()
   if name in PROPERTIES:
    element.set(name,value);count+=1
   else:remaining.append(f'{name}: {value}')
  if remaining:element.set('style','; '.join(remaining))
  else:element.attrib.pop('style',None)
 tree.write(path,encoding='utf-8',xml_declaration=True)
 return count

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('paths',nargs='+',type=Path);args=p.parse_args()
 for path in args.paths:print(path.name,normalize(path))
