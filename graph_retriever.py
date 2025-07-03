from neo4j import GraphDatabase
from typing import List, Dict

class GraphRetriever:
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        """Connect to Neo4j and setup sample graph."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.setup_sample_graph()

    def close(self):
        """Close Neo4j connection."""
        self.driver.close()

    def setup_sample_graph(self):
        """Create nodes and relationships in Neo4j for testing."""
        queries = [
            "MATCH (n) DETACH DELETE n",  # clear all nodes

            # Customers
            "CREATE (:Customer {id: 3, name: 'BioMed Research', revenue: 12000000, risk_level: 'HIGH'})",
            "CREATE (:Customer {id: 6, name: 'NextGen Pharmaceuticals', revenue: 45000000, risk_level: 'HIGH'})",
            "CREATE (:Customer {id: 9, name: 'Digital Health Corp', revenue: 18000000, risk_level: 'HIGH'})",

            # Facilities
            "CREATE (:Facility {id: 'FAC-004', name: 'Factory Delta', location: 'Sacramento', emissions: 2100})",
            "CREATE (:Facility {id: 'FAC-001', name: 'Plant Alpha', location: 'San Jose', emissions: 1250})",

            # Regulations
            "CREATE (:Regulation {type: 'CO2_Limit', value: 2000, unit: 'tons'})",
            "CREATE (:Regulation {type: 'FDA_Approval', status: 'pending'})",

            # Molecule
            "CREATE (:Molecule {name: 'Molecule X', phase: 'II', response_rate: 0.67})",

            # Industries
            "UNWIND ['Biotech', 'Pharma', 'Energy'] AS name CREATE (:Industry {name: name})",

            # Relationships
            "MATCH (c:Customer {id: 3}), (i:Industry {name: 'Biotech'}) CREATE (c)-[:OPERATES_IN]->(i)",
            "MATCH (c:Customer {id: 6}), (i:Industry {name: 'Pharma'}) CREATE (c)-[:OPERATES_IN]->(i)",
            "MATCH (c:Customer {id: 6}), (r:Regulation {type: 'FDA_Approval'}) CREATE (c)-[:SUBJECT_TO]->(r)",
            "MATCH (f:Facility {id: 'FAC-004'}), (r:Regulation {type: 'CO2_Limit'}) CREATE (f)-[:EXCEEDS]->(r)",
            "MATCH (m:Molecule {name: 'Molecule X'}), (i:Industry {name: 'Biotech'}) CREATE (m)-[:TARGETS]->(i)",
            "MATCH (c:Customer {id: 3}), (m:Molecule {name: 'Molecule X'}) CREATE (c)-[:RESEARCHES]->(m)"
        ]

        with self.driver.session() as session:
            for q in queries:
                session.run(q)
        print("Sample graph loaded.")

    def query(self, cypher: str, params: Dict = {}) -> List[Dict]:
        """Run a Cypher query and return results as list of dicts."""
        try:
            with self.driver.session() as session:
                result = session.run(cypher, **params)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Query failed: {e}")
            return []

    def find_compliance_violations(self) -> List[Dict]:
        return self.query("""
        MATCH (f:Facility)-[:EXCEEDS]->(r:Regulation)
        RETURN f.name AS facility, f.location AS location, 
               f.emissions AS emissions, r.value AS limit
        """)

    def find_high_risk_customers_in_industry(self, industry: str) -> List[Dict]:
        return self.query("""
        MATCH (c:Customer)-[:OPERATES_IN]->(i:Industry {name: $industry})
        WHERE c.risk_level = 'HIGH'
        RETURN c.name AS customer, c.revenue AS revenue, c.risk_level AS risk
        """, {"industry": industry})


# ---- Test block ----
if __name__ == "__main__":
    retriever = GraphRetriever()

    print("\nHigh-Risk Customers in Biotech:")
    for row in retriever.find_high_risk_customers_in_industry("Biotech"):
        print(row)

    print("\nCompliance Violations:")
    for row in retriever.find_compliance_violations():
        print(row)

    retriever.close()
