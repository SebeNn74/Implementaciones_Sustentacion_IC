from TestSuite import run_tests

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                  IMPLEMENTACIÓN DBSCAN DESDE CERO                        ║
    ║  Algoritmo: Density-Based Spatial Clustering of Applications with Noise  ║    
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    print("""Características:                                                      
    • Detecta clusters de forma arbitraria                              
    • Identifica outliers automáticamente                               
    • No requiere especificar número de clusters                        
    • Robusto ante ruido   
    """)

    # Ejecutar suite de pruebas
    run_tests()
