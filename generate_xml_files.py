import xml.etree.ElementTree as ET
from random import randint

def generate_xml_files():
    catalog_root = ET.Element("catalog")

    for i in range(1, 11):
        product = ET.SubElement(catalog_root, "product")
        sku = ET.SubElement(product, "sku")
        name = ET.SubElement(product, "name")
        price = ET.SubElement(product, "price")

        sku.text = f"SKU{i}"
        name.text = f"Product {i}"
        price.text = str(randint(10, 100))

    catalog_tree = ET.ElementTree(catalog_root)
    catalog_tree.write("catalog.xml")

    descriptions_root = ET.Element("descriptions")

    for i in range(1, 11):
        description = ET.SubElement(descriptions_root, "description")
        sku = ET.SubElement(description, "sku")
        text = ET.SubElement(description, "text")

        sku.text = f"SKU{i}"
        text.text = f"Description for Product {i}"

    descriptions_tree = ET.ElementTree(descriptions_root)
    descriptions_tree.write("descriptions.xml")

generate_xml_files()
