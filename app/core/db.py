from fastapi import Depends
from sqlalchemy.util.typing import Annotated
from app.core.settings import settings
from opensearchpy import OpenSearch
from sqlmodel import Field, Session, SQLModel, create_engine


# Create the client with SSL/TLS enabled, but hostname verification disabled.
open_search_client = OpenSearch(
    hosts = [{'host': settings.OPENSEARCH_HOST, 'port': settings.OPENSEARCH_PORT}],
    http_compress = True, # enables gzip compression for request bodies
    use_ssl = True,
    verify_certs = False,
    ssl_assert_hostname = False,
    ssl_show_warn = False
)

# Create the Primary (Write) Database Engine
write_engine = create_engine(
    settings.sqlalchemy_database_uri,
    echo=False, # Set to False in production
    pool_pre_ping=True
)

def get_pgvector_session():
    with Session(write_engine) as session:
        yield session
        
def get_open_search_session():
    """ Returing opensearch client object"""
    return open_search_client

# Creating an instance of the OpenSearchDep for fastapi dependency injection
OpenSearchDep = Annotated[OpenSearch, Depends(get_open_search_session)]
