import os
import time
import pytest
import threading
import pandasai as pd

from pandasai.data_loader.duck_db_connection_manager import DuckDBConnectionManager


class TestDuckDBConnectionManager:
    @pytest.fixture
    def duck_db_manager(self):
        manager = DuckDBConnectionManager()
        yield manager
        manager.close()

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    def test_temp_file_creation_and_deletion(self, duck_db_manager):
        """Test that temporary db file is created and deleted properly"""
        # Get the db file path
        db_file = duck_db_manager._db_file
        
        # Verify file exists while manager is active
        assert os.path.exists(db_file)
        
        # Close manager and verify file is deleted
        duck_db_manager.close()
        assert not os.path.exists(db_file)

    def test_connection_pool_exhaustion(self, duck_db_manager):
        """Test that connection requests timeout properly when pool is exhausted"""
        # Get all connections to exhaust the pool
        connections = []
        for _ in range(duck_db_manager._pool_size):
            connections.append(duck_db_manager._get_connection())
        
        # Test that new request times out after _max_wait_time
        start_time = time.time()
        with pytest.raises(RuntimeError, match="No available connections in the pool"):
            duck_db_manager._get_connection()
        elapsed_time = time.time() - start_time
        
        # Verify timeout is approximately _max_wait_time
        assert abs(elapsed_time - duck_db_manager._max_wait_time) < 0.5
        
        # Release connections back to pool
        for conn in connections:
            duck_db_manager._release_connection(conn)

    def test_concurrent_access_thread_safety(self, duck_db_manager, sample_df):
        """Test thread safety with concurrent access"""
        num_threads = 10
        results = []
        errors = []
        
        def worker():
            try:
                # Register a unique table name per thread
                table_name = f"table_{threading.get_ident()}"
                duck_db_manager.register(table_name, sample_df)
                
                # Execute a query
                result = duck_db_manager.sql(f"SELECT * FROM {table_name}")
                results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        # Create and start threads
        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Verify no errors occurred
        assert not errors
        assert len(results) == num_threads
        
        # Verify all results are correct
        for result in results:
            assert len(result) == 3  # Should match sample_df row count

    def test_table_registration_thread_safety(self, duck_db_manager, sample_df):
        """Test thread safety of table registration and SQL operations with high concurrency"""
        num_threads = 50
        table_name = "shared_table"
        results = []
        errors = []
        
        def worker(thread_id):
            try:
                # Register the same table from multiple threads
                duck_db_manager.register(table_name, sample_df)
                
                # Test various SQL operations
                # 1. Simple select
                select_result = duck_db_manager.sql(f"SELECT * FROM {table_name}")
                assert len(list(select_result.fetchall())) == 3
                
                # 2. Count query - convert to DataFrame first
                count_result = duck_db_manager.sql(f"SELECT COUNT(*) FROM {table_name}").df()
                assert count_result.iloc[0, 0] == 3
                
                # 3. Conditional query
                cond_result = duck_db_manager.sql(
                    f"SELECT col2 FROM {table_name} WHERE col1 = {thread_id % 3 + 1}"
                )
                assert len(list(cond_result.fetchall())) == 1
                
                # 4. Aggregation - convert to DataFrame first
                agg_result = duck_db_manager.sql(f"SELECT SUM(col1) FROM {table_name}").df()
                assert agg_result.iloc[0, 0] == 6
                
                results.append(True)
            except Exception as e:
                errors.append(f"Thread {thread_id} failed: {str(e)}")
        
        # Create and start threads
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join(timeout=10)  # Add timeout to prevent hanging
        
        # Verify no errors occurred
        assert not errors, f"Errors occurred in threads: {errors}"
        assert len(results) == num_threads, "Not all threads completed successfully"
        
        # Final verification of table integrity
        final_result = duck_db_manager.sql(f"SELECT * FROM {table_name}")
        assert len(final_result) == 3, "Table data corrupted"

    def test_connection_correct_closing_doesnt_throw(self, duck_db_manager):
        """Test that closing connections doesn't throw exceptions"""
        duck_db_manager.close()
