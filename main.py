
from contextlib import asynccontextmanager
from elasticsearch import Elasticsearch
from fastapi import FastAPI, HTTPException, UploadFile
import xml.etree.ElementTree as ET
from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import joblib

logger = logging.getLogger(__name__)

# Загрузка и считывание данных из XML файлов
def load_data_from_xml(catalog_xml, descriptions_xml):
    catalog_tree = ET.parse(catalog_xml)
    catalog_root = catalog_tree.getroot()

    descriptions_tree = ET.parse(descriptions_xml)
    descriptions_root = descriptions_tree.getroot()

    return catalog_root, descriptions_root

# Сопоставление описаний с товарами используя SKU
def match_descriptions_with_products(catalog, descriptions):
    data_dict = {}

    for description in descriptions:
        sku = description.find('sku').text
        description_text = description.find('text').text

        if sku in data_dict:
            data_dict[sku]['description'] = description_text
        else:
            data_dict[sku] = {'description': description_text}

    for product in catalog:
        sku = product.find('sku').text

        if sku in data_dict:
            data_dict[sku]['product_name'] = product.find('name').text
            data_dict[sku]['price'] = float(product.find('price').text)

    return data_dict

# Установка и настройка библиотеки ElasticSearch
def setup_elasticsearch():

    es = Elasticsearch(['http://localhost:9200'])
    

    mapping = {
        "mappings": {
            "properties": {
                "product_name": {
                    "type": "text" 
                },
                "description": {
                    "type": "text"
                },
                "price": {
                    "type": "float"
                },
                "vector": {
                    "type": "dense_vector",
                    # "dims": 768
                }
            }
        }
    }
    
    index_name = 'products'

    es.indices.create(index=index_name, body=mapping, ignore=400)
    
    return es, index_name

# Сохранение данных в ElasticSearch
def save_data_to_elasticsearch(es, index_name, data_dict):
    for sku, data in data_dict.items():
        es.index(index=index_name, id=sku, body=data)

# Создание векторов записей
def create_vectors(data_dict):
    
    descriptions = [data['description'] for data in data_dict.values()]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(descriptions).toarray()
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    
    return vectors

# Сохранение векторов в ElasticSearch
def save_vectors_to_elasticsearch(es, index_name, data_dict, vectors):
    for i, (sku, data) in enumerate(data_dict.items()):
        data['vector'] = vectors[i].tolist()
        es.index(index=index_name, id=sku, body=data)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the data
    on_startup()
    
    yield
    
    # Clean up the resources
    on_shutdown()


app = FastAPI(title='Semantic Search API', lifespan=lifespan)

def on_startup():

    # Загрузка данных
    catalog, descriptions = load_data_from_xml('catalog.xml', 'descriptions.xml')

    # Сопоставление описаний с товарами
    data_dict = match_descriptions_with_products(catalog, descriptions)

    # Установка и настройка ElasticSearch
    es, index_name = setup_elasticsearch()
    logger.info("Successfully connected to ElasticSearch")

    # Сохранение данных в ElasticSearch
    save_data_to_elasticsearch(es, index_name, data_dict)

    # Создание векторов записей
    vectors = create_vectors(data_dict)

    # Сохранение векторов в ElasticSearch
    save_vectors_to_elasticsearch(es, index_name, data_dict, vectors)
    logger.info("Successfully save data to ElasticSearch")


def on_shutdown():
    logger.info("Shutting down API")


@app.get("/")
async def root():
    #main()
    return {"message": "Hello World"}


@app.post("/search", tags=["Search"], response_model=list[dict])
async def search(query_text: str):

    query_vector = vectorize_text(query_text)
    
    es, index_name = setup_elasticsearch()
    
    vector_results = search_by_vectors(es, index_name, query_vector)

    return vector_results


def vectorize_text(query_text):
    # Создаем экземпляр векторизатора TF-IDF
    # vectorizer = TfidfVectorizer()

    vectorizer = joblib.load('tfidf_vectorizer.pkl')

    # Преобразуем текст запроса в вектор
    query_vector = vectorizer.transform([query_text]).toarray()[0]

    return query_vector


def search_by_vectors(es, index_name, query_vector):

    # Поиск ближайших соседей по вектору запроса с использованием ElasticSearch
    script_query =  {
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.queryVector, 'vector') + 1.0",
                    "params": {"queryVector": query_vector}
                }
            }
        }
    }
    
    top_k = 5
    try: 
        response = es.search(
            index=index_name, 
            size=top_k,
            body=script_query,
        )
        print("Got %d Hits:" % response['hits']['total']['value'])
        for hit in response['hits']['hits']:
            print("%(product_name)s %(description)s: %(price)s" % hit["_source"])

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))        
    
    if response:
        return response["hits"]["hits"]
    return None    
    



