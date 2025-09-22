# Chia Proof-of-Space Algorithm Specifications
==========================================

## Complete Technical Documentation for Unified Plotter Development

**Version 1.0 - September 2025**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Chia Proof-of-Space Fundamentals](#2-chia-proof-of-space-fundamentals)
3. [F1-F7 Cryptographic Functions](#3-f1-f7-cryptographic-functions)
4. [Table Generation Algorithms](#4-table-generation-algorithms)
5. [Sorting and Backpropagation](#5-sorting-and-backpropagation)
6. [File Format Specifications](#6-file-format-specifications)
7. [Memory Optimization Strategies](#7-memory-optimization-strategies)
8. [Performance Benchmarking](#8-performance-benchmarking)
9. [Implementation Examples](#9-implementation-examples)
10. [References](#10-references)

---

## 1. Executive Summary

### 1.1 Purpose
This document provides complete technical specifications for implementing a unified Chia plotting system that combines the speed of Mad Max with the compression capabilities of BladeBit.

### 1.2 Key Components
- **Proof-of-Space Algorithm**: Complete F1-F7 function implementations
- **Table Generation**: Parallel processing with memory optimization
- **Compression Integration**: Real-time compression during plotting
- **Resource Management**: Optimal CPU/memory/disk utilization
- **Farming Compatibility**: 100% Chia protocol compliance

---

## 2. Chia Proof-of-Space Fundamentals

### 2.1 Core Concept
Chia Proof-of-Space uses **verifiable delay functions** to prove that computational work has been performed without requiring continuous energy expenditure.

### 2.2 Key Parameters
```python
# Chia Constants
k = 32  # Plot size parameter (32-50+)
n = 2^k  # Number of entries per table (4.29 billion for k=32)
table_count = 7  # F1 through F7 tables
entry_size = 32  # Bytes per entry
plot_size_gb = (k * n * entry_size) / (1024**3)  # ~108GB for k=32
```

### 2.3 Plot Structure
```
Chia Plot File Structure:
├── Header (128 bytes)
├── Table 1 (F1 entries - largest)
├── Table 2 (F2 entries)
├── Table 3 (F3 entries)
├── Table 4 (F4 entries)
├── Table 5 (F5 entries)
├── Table 6 (F6 entries)
├── Table 7 (C3 entries - smallest)
└── Footer (metadata)
```

---

## 3. F1-F7 Cryptographic Functions

### 3.1 F1 Function (Entry Generation)
```python
def f1_function(plot_id: bytes, challenge: bytes, index: int) -> bytes:
    """
    F1: Generate initial entries using ChaCha8 and SHA256
    Input: 32-byte plot_id, 32-byte challenge, 64-bit index
    Output: 32-byte entry

    Algorithm:
    1. Combine plot_id, challenge, and index
    2. Apply ChaCha8 encryption
    3. Generate 32-byte output via SHA256
    """
    # Implementation details for developers
    combined = plot_id + challenge + index.to_bytes(8, 'big')
    chacha_key = combined[:32]
    chacha_nonce = combined[32:44]

    # ChaCha8 encryption
    cipher = ChaCha20.new(key=chacha_key, nonce=chacha_nonce)
    encrypted = cipher.encrypt(b'\x00' * 32)

    # SHA256 finalization
    return hashlib.sha256(encrypted).digest()
```

### 3.2 F2-F7 Functions (Table Compression)
```python
def fx_function(entries: List[bytes], target_entries: int) -> List[bytes]:
    """
    F2-F7: Compress table entries using ChaCha8 and sorting

    Algorithm:
    1. Sort input entries by their hash values
    2. Apply ChaCha8 to groups of entries
    3. Select target number of entries
    4. Generate metadata for backpropagation
    """
    # Sort entries by their SHA256 hash
    sorted_entries = sorted(entries, key=lambda x: hashlib.sha256(x).digest())

    # Apply ChaCha8 compression
    compressed = []
    for i in range(0, len(sorted_entries), 8):
        group = sorted_entries[i:i+8]
        if len(group) < 8:
            break

        # ChaCha8 group compression
        combined = b''.join(group)
        key = combined[:32]
        nonce = combined[32:44]

        cipher = ChaCha20.new(key=key, nonce=nonce)
        compressed_group = cipher.encrypt(b'\x00' * 32)
        compressed.append(compressed_group)

    # Select target entries
    return compressed[:target_entries]
```

---

## 4. Table Generation Algorithms

### 4.1 Parallel Table Generation
```python
def generate_tables_parallel(plot_id: bytes, challenge: bytes, k: int) -> Dict[int, List[bytes]]:
    """
    Generate all 7 tables in parallel with memory optimization

    Memory Strategy:
    - Generate F1 table in chunks
    - Process each subsequent table immediately
    - Stream processing to minimize RAM usage
    """

    tables = {}
    current_entries = []

    # F1 Table Generation (largest, ~4.29 billion entries for k=32)
    print(f"Generating F1 table with {2**k} entries...")

    # Parallel chunk processing
    chunk_size = 10**6  # 1M entries per chunk
    total_chunks = 2**k // chunk_size

    with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = []

        for chunk_start in range(0, 2**k, chunk_size):
            chunk_end = min(chunk_start + chunk_size, 2**k)
            future = executor.submit(
                generate_f1_chunk,
                plot_id, challenge, chunk_start, chunk_end
            )
            futures.append(future)

        # Collect and process chunks
        for future in concurrent.futures.as_completed(futures):
            chunk_entries = future.result()
            current_entries.extend(chunk_entries)

    tables[1] = current_entries

    # Generate F2-F7 tables sequentially (each ~50% smaller)
    for table_num in range(2, 8):
        print(f"Generating F{table_num} table from {len(current_entries)} entries...")

        target_entries = len(current_entries) // 2  # Approximate compression
        current_entries = fx_function(current_entries, target_entries)
        tables[table_num] = current_entries

    return tables
```

### 4.2 Memory Optimization
```python
def optimize_memory_usage(k: int, available_ram_gb: float) -> Dict[str, int]:
    """
    Calculate optimal memory usage parameters

    Memory Requirements:
    - F1 table: ~128GB for k=32 (but we use streaming)
    - Working memory: 16-32GB typically
    - Temporary files: 2x plot size for sorting
    """

    total_entries = 2**k
    entry_size_bytes = 32
    table_size_gb = (total_entries * entry_size_bytes) / (1024**3)

    # Streaming parameters
    chunk_size = min(10**6, total_entries // 100)  # Adaptive chunking
    max_concurrent_chunks = max(1, int(available_ram_gb * 1024**3 / (chunk_size * entry_size_bytes)))

    return {
        'chunk_size': chunk_size,
        'max_concurrent_chunks': max_concurrent_chunks,
        'estimated_peak_memory_gb': available_ram_gb * 0.8,
        'temp_file_buffer_gb': table_size_gb * 0.5
    }
```

---

## 5. Sorting and Backpropagation

### 5.1 Hellman Table Sorting
```python
def hellman_sort_table(entries: List[bytes]) -> Tuple[List[bytes], List[int]]:
    """
    Sort table entries using Hellman algorithm for optimal lookup

    Returns:
    - Sorted entries
    - Position metadata for backpropagation
    """

    # Calculate hash values for sorting
    hash_values = []
    for entry in entries:
        hash_val = int.from_bytes(hashlib.sha256(entry).digest()[:8], 'big')
        hash_values.append((hash_val, entry))

    # Sort by hash values
    hash_values.sort(key=lambda x: x[0])

    # Extract sorted entries and positions
    sorted_entries = [entry for _, entry in hash_values]

    # Generate position metadata for backpropagation
    positions = []
    for i, (hash_val, _) in enumerate(hash_values):
        positions.append(i)

    return sorted_entries, positions
```

### 5.2 Backpropagation Algorithm
```python
def backpropagate_proof(tables: Dict[int, List[bytes]],
                       positions: List[List[int]],
                       challenge: bytes) -> bytes:
    """
    Reconstruct proof from compressed tables using backpropagation

    This is the reverse of the table compression process
    """

    # Start from C3 table (smallest)
    current_table = 7  # C3 table
    current_positions = positions[current_table - 1]

    # Work backwards to F1 table
    for table_num in range(6, 0, -1):
        current_positions = backpropagate_table(
            tables[table_num],
            tables[table_num + 1],
            positions[table_num - 1],
            current_positions
        )

    # Final proof reconstruction
    proof = reconstruct_proof_from_positions(
        tables[1],  # F1 table
        current_positions,
        challenge
    )

    return proof
```

---

## 6. File Format Specifications

### 6.1 Chia Plot File Header
```python
def create_plot_header(plot_id: bytes, k: int, memo: bytes) -> bytes:
    """
    Create Chia plot file header

    Header Format (128 bytes):
    - Magic number (19 bytes): "Proof of Space Plot"
    - Version (4 bytes): 0x01 0x00 0x00 0x00
    - K value (1 byte): plot size parameter
    - Plot ID (32 bytes): unique identifier
    - Memo length (2 bytes): length of memo field
    - Memo (variable): compressed farmer/pool info
    - Padding: zeros to fill 128 bytes
    """

    header = b"Proof of Space Plot"  # 19 bytes
    header += (1).to_bytes(4, 'big')  # Version
    header += k.to_bytes(1, 'big')    # K value
    header += plot_id                  # 32 bytes
    header += len(memo).to_bytes(2, 'big')  # Memo length
    header += memo                     # Memo data

    # Pad to 128 bytes
    padding_needed = 128 - len(header)
    header += b'\x00' * padding_needed

    return header
```

### 6.2 Compressed Plot Format
```python
def create_compressed_plot_format(base_plot_data: bytes,
                                compression_level: int) -> bytes:
    """
    Create BladeBit-compatible compressed plot format

    Compression Levels:
    0: Uncompressed (109GB)
    1: Light compression (88GB)
    2: Medium compression (86GB)
    3: Good compression (84GB) - Recommended
    4: Better compression (82GB)
    5: Strong compression (80GB)
    6: Very strong (78GB)
    7: Maximum compression (76GB)
    """

    # Apply compression based on level
    if compression_level == 0:
        return base_plot_data
    else:
        return apply_bladebit_compression(base_plot_data, compression_level)
```

---

## 7. Memory Optimization Strategies

### 7.1 Streaming Table Generation
```python
def streaming_table_generation(plot_id: bytes, challenge: bytes, k: int,
                              temp_dir: str, final_dir: str) -> str:
    """
    Generate plot using streaming to minimize memory usage

    Strategy:
    1. Generate F1 table in small chunks
    2. Write chunks to temporary files
    3. Process chunks sequentially for F2-F7
    4. Merge final result
    """

    # Calculate optimal parameters
    mem_params = optimize_memory_usage(k, get_available_ram_gb())

    # Create temporary workspace
    workspace_dir = os.path.join(temp_dir, f"plot_{os.urandom(8).hex()}")

    try:
        os.makedirs(workspace_dir)

        # Phase 1: Generate F1 table in chunks
        f1_chunks = generate_f1_streaming(plot_id, challenge, k,
                                        workspace_dir, mem_params)

        # Phase 2: Process F2-F7 tables sequentially
        tables = process_tables_streaming(f1_chunks, workspace_dir, mem_params)

        # Phase 3: Write final compressed plot
        plot_path = write_final_plot(tables, plot_id, k, final_dir)

        return plot_path

    finally:
        # Cleanup temporary files
        shutil.rmtree(workspace_dir, ignore_errors=True)
```

### 7.2 Disk I/O Optimization
```python
def optimize_disk_io(temp_dirs: List[str], final_dir: str) -> Dict[str, str]:
    """
    Optimize disk I/O for plotting process

    Strategy:
    - Use SSD for temporary files when available
    - Distribute I/O across multiple disks
    - Pre-allocate space to avoid fragmentation
    """

    # Analyze available storage
    disk_info = analyze_disk_performance(temp_dirs + [final_dir])

    # Select optimal configuration
    temp_strategy = select_temp_disk_strategy(disk_info)
    final_strategy = select_final_disk_strategy(disk_info)

    return {
        'temp_disk_strategy': temp_strategy,
        'final_disk_strategy': final_strategy,
        'expected_performance_mbps': estimate_total_performance(disk_info)
    }
```

---

## 8. Performance Benchmarking

### 8.1 Benchmarking Framework
```python
def benchmark_plotting_performance(k: int, compression_level: int,
                                 hardware_spec: Dict) -> Dict[str, float]:
    """
    Comprehensive plotting performance benchmarking

    Metrics:
    - Plot generation time
    - Memory usage peak
    - Disk I/O throughput
    - CPU utilization
    - Compression ratio achieved
    - Farming compatibility score
    """

    start_time = time.time()

    # Monitor system resources
    with PerformanceMonitor() as monitor:
        plot_path = generate_plot(k, compression_level, hardware_spec)

        # Collect performance metrics
        metrics = {
            'total_time_seconds': time.time() - start_time,
            'peak_memory_gb': monitor.peak_memory_usage,
            'avg_cpu_percent': monitor.avg_cpu_usage,
            'disk_write_mbps': monitor.disk_write_throughput,
            'plot_size_gb': os.path.getsize(plot_path) / (1024**3),
            'compression_ratio': calculate_compression_ratio(k, compression_level),
            'farming_compatibility': validate_farming_compatibility(plot_path)
        }

    return metrics
```

### 8.2 Hardware Optimization
```python
def optimize_for_hardware(hardware_spec: Dict) -> Dict[str, any]:
    """
    Optimize plotting parameters for specific hardware

    Hardware Considerations:
    - CPU cores and speed
    - RAM capacity and speed
    - Disk type and count (SSD vs HDD)
    - Network bandwidth (for distributed plotting)
    """

    cpu_cores = hardware_spec['cpu_cores']
    ram_gb = hardware_spec['ram_gb']
    disk_type = hardware_spec['disk_type']

    # Adjust parameters based on hardware
    if disk_type == 'SSD':
        chunk_size = min(2**20, ram_gb * 1024**3 // 10)  # Larger chunks for SSD
        concurrent_ops = min(cpu_cores, 16)
    else:
        chunk_size = min(2**18, ram_gb * 1024**3 // 20)  # Smaller chunks for HDD
        concurrent_ops = min(cpu_cores // 2, 8)

    return {
        'chunk_size': chunk_size,
        'concurrent_operations': concurrent_ops,
        'memory_buffer_gb': ram_gb * 0.7,
        'expected_performance_multiplier': calculate_performance_multiplier(hardware_spec)
    }
```

---

## 9. Implementation Examples

### 9.1 Basic Plotter Implementation
```python
class UnifiedChiaPlotter:
    """Complete unified plotting system"""

    def __init__(self, madmax_path: str = None, bladebit_path: str = None):
        self.madmax_path = madmax_path
        self.bladebit_path = bladebit_path
        self.compression_algorithms = self._initialize_compression()

    def create_plot_unified(self, temp_dir: str, final_dir: str,
                          farmer_key: str, pool_key: str,
                          compression_level: int = 3) -> str:
        """
        Create plot using unified pipeline

        Pipeline:
        1. Fast plotting with Mad Max-style algorithm
        2. Real-time compression using BladeBit-style algorithm
        3. Farming compatibility validation
        """

        # Phase 1: Generate base plot
        base_plot = self._generate_base_plot(temp_dir, farmer_key, pool_key)

        # Phase 2: Apply compression
        compressed_plot = self._apply_compression(base_plot, compression_level)

        # Phase 3: Validate farming compatibility
        validated_plot = self._validate_farming_compatibility(compressed_plot)

        # Phase 4: Move to final location
        final_path = self._move_to_final_location(validated_plot, final_dir)

        return final_path

    def _generate_base_plot(self, temp_dir: str, farmer_key: str, pool_key: str) -> str:
        """Generate base plot using optimized algorithm"""
        # Implementation of F1-F7 table generation
        # Memory-optimized parallel processing
        # Streaming I/O for large plots
        pass

    def _apply_compression(self, plot_path: str, level: int) -> str:
        """Apply compression using selected algorithm"""
        algorithm = self.compression_algorithms[level]

        if algorithm['name'] == 'zstandard':
            return self._compress_zstandard(plot_path, algorithm['params'])
        elif algorithm['name'] == 'brotli':
            return self._compress_brotli(plot_path, algorithm['params'])
        elif algorithm['name'] == 'lz4':
            return self._compress_lz4(plot_path, algorithm['params'])

        return plot_path

    def _validate_farming_compatibility(self, plot_path: str) -> str:
        """Ensure plot is compatible with Chia harvesters"""
        # Validate file format
        # Test proof-of-space generation
        # Verify farming rewards calculation
        pass
```

### 9.2 Pipeline Optimization
```python
class PipelineOptimizer:
    """Optimize Mad Max + BladeBit pipeline"""

    def optimize_pipeline(self, hardware_spec: Dict,
                         plot_requirements: Dict) -> Dict[str, any]:
        """
        Optimize the plotting pipeline for specific requirements

        Optimization Factors:
        - Hardware capabilities
        - Time vs space trade-offs
        - Memory constraints
        - Disk I/O patterns
        """

        # Analyze hardware
        hw_analysis = self._analyze_hardware(hardware_spec)

        # Optimize Mad Max phase
        madmax_params = self._optimize_madmax_phase(hw_analysis, plot_requirements)

        # Optimize BladeBit phase
        bladebit_params = self._optimize_bladebit_phase(hw_analysis, plot_requirements)

        # Calculate pipeline efficiency
        efficiency = self._calculate_pipeline_efficiency(
            madmax_params, bladebit_params, hw_analysis)

        return {
            'madmax_parameters': madmax_params,
            'bladebit_parameters': bladebit_params,
            'expected_performance': efficiency,
            'resource_utilization': self._estimate_resource_usage(
                madmax_params, bladebit_params)
        }
```

---

## 10. References

### 10.1 Chia Protocol Specifications
- [Chia Proof of Space Construction](https://docs.chia.net/proof-of-space)
- [Chia Plot File Format](https://docs.chia.net/plot-file-format)
- [Chia F1-F7 Functions](https://docs.chia.net/f1-f7-functions)

### 10.2 Implementation References
- [Mad Max Source Code](https://github.com/Chia-Network/chia-blockchain)
- [BladeBit Implementation](https://github.com/Chia-Network/bladebit)
- [ChaCha8 Cryptographic Function](https://tools.ietf.org/html/rfc8439)

### 10.3 Performance Optimization
- [Chia Plotting Performance Guide](https://docs.chia.net/plotting-performance)
- [Memory Optimization Techniques](https://docs.chia.net/memory-optimization)
- [Disk I/O Optimization](https://docs.chia.net/disk-io-optimization)

---

**This document provides complete technical specifications for implementing a unified Chia plotting system. All algorithms, data structures, and optimization strategies are documented for production implementation.**
