# 🌱 Chia Resources Database

A comprehensive collection of Chia Network documentation, APIs, tools, and community resources scraped and organized for easy access and integration.

## 📁 Database Structure

```
chia_resources/
├── chia_resources_database.json    # Main database file
├── resources_index.json           # Quick lookup index
├── chia_resources_report.txt      # Human-readable report
├── chia_resource_scraper.py       # Scraping tool
├── chia_resource_query.py         # Query interface
└── README.md                      # This file
```

## 🗂️ Resource Categories

### 📚 Documentation (5 resources)
- **chia_main_docs**: Main Chia documentation site
- **chialisp_docs**: Chialisp programming language documentation
- **dev_guides**: Developer guides and tutorials
- **rpc_reference**: RPC API reference documentation
- **cli_reference**: CLI command reference documentation

### 🔌 APIs (4 resources)
- **spacescan_api**: Blockchain explorer API
- **mintgarden_api**: NFT marketplace API
- **dexie_api**: DEX aggregator API
- **xch_network_dev**: XCH network developer resources

### 💻 GitHub (1 resource)
- **chia_network_org**: Chia Network GitHub organization with all repositories

### 👥 Community (2 resources)
- **discord**: Official Chia Discord server
- **spacescan**: Blockchain explorer website

## 🛠️ Usage

### Command Line Interface

```bash
# Show database statistics
python chia_resource_query.py --stats

# Search for resources
python chia_resource_query.py --search "wallet"

# List resources in category
python chia_resource_query.py --category docs

# Get detailed info about specific resource
python chia_resource_query.py --info chia_main_docs

# Search by topic
python chia_resource_query.py --topic farming

# Interactive query mode
python chia_resource_query.py --interactive
```

### Interactive Mode

```bash
python chia_resource_query.py --interactive

chia-query> help
chia-query> list
chia-query> category docs
chia-query> search wallet
chia-query> info chia_main_docs
chia-query> topic farming
chia-query> apis
chia-query> repos
chia-query> stats
chia-query> quit
```

### Python API

```python
from chia_resource_query import ChiaResourceQuery

query = ChiaResourceQuery()

# Get all documentation resources
docs = query.query_resources(category='docs')

# Search for wallet-related resources
wallets = query.query_resources(search_term='wallet')

# Get detailed info about specific resource
info = query.get_resource_info('chia_main_docs')

# Get API endpoints
apis = query.get_api_endpoints()

# Get GitHub repositories
repos = query.get_github_repositories()

# Export resource data
json_data = query.export_resource_data('chia_main_docs', 'json')
text_data = query.export_resource_data('chia_main_docs', 'text')
```

## 🔍 Search Capabilities

### Text Search
Search through all resource content including:
- Titles and descriptions
- Documentation sections
- API endpoint descriptions
- Code examples
- URLs and links

### Topic-Based Search
Pre-defined topic categories:
- **wallet**: Wallet management, keys, addresses, balances
- **farming**: Farming, plotting, harvesting, rewards
- **development**: APIs, RPC, SDKs, libraries
- **trading**: DEX, offers, markets, exchanges
- **nft**: NFTs, tokens, collections, metadata
- **staking**: Staking, pooling, delegation
- **mining**: Mining, proof-of-space, plotting

### Category Search
Search within specific categories:
- `docs`: Documentation resources
- `apis`: API documentation
- `github`: GitHub repositories
- `community`: Community resources

## 📊 Database Features

### Comprehensive Data Collection
- **Structured metadata**: Title, description, URL, type
- **Content extraction**: Headings, paragraphs, links
- **API documentation**: Endpoints, parameters, examples
- **Code examples**: Extracted from documentation
- **Navigation structure**: Site navigation and sections

### Search and Filtering
- **Full-text search**: Across all content fields
- **Category filtering**: Search within specific categories
- **Type filtering**: Filter by resource type
- **Relevance ranking**: Results ranked by relevance score

### Export Capabilities
- **JSON format**: Complete structured data
- **Text format**: Human-readable summaries
- **API endpoints**: Extracted API endpoint information
- **GitHub repos**: Repository information and descriptions

## 🔄 Scraping and Updates

### Running the Scraper

```bash
cd chia_resources
python chia_resource_scraper.py
```

This will:
1. Scrape all configured Chia resources
2. Extract structured information
3. Update the database
4. Generate reports

### Configured Resources

The scraper is configured to collect data from these Chia Network resources:

```python
chia_resources = {
    "docs": {
        "chia_main_docs": "https://docs.chia.net/",
        "chialisp_docs": "https://chialisp.com/",
        "dev_guides": "https://docs.chia.net/dev-guides-home/",
        "rpc_reference": "https://docs.chia.net/reference-client/rpc-reference/rpc/",
        "cli_reference": "https://docs.chia.net/reference-client/cli-reference/cli/"
    },
    "apis": {
        "spacescan_api": "https://docs.spacescan.io/api/address/xch_balance/",
        "mintgarden_api": "https://api.mintgarden.io/docs",
        "dexie_api": "https://dexie.space/api",
        "xch_network_dev": "https://xch.network/developers/"
    },
    "github": {
        "chia_network_org": "https://github.com/chia-network"
    },
    "community": {
        "discord": "https://discord.com/invite/chia",
        "spacescan": "https://www.spacescan.io/"
    }
}
```

## 🎯 Integration Opportunities

### With GIF-VM
- **Encode Chia docs as executable GIFs**
- **Visual programming for Chia blockchain operations**
- **Evolutionary optimization of Chia farming strategies**

### With Advanced Math Frameworks
- **CUDNT**: Optimize Chia plot compression algorithms
- **EIMF**: Energy-aware Chia farming optimization
- **CHAIOS**: Consciousness-guided Chia network decisions

### With SquashPlot
- **Integrate Chia APIs for real-time data**
- **Use Chia documentation for farming optimization**
- **GitHub integration for automated updates**

## 📈 Database Statistics

- **Total Resources**: 12
- **Categories**: 7 (docs, apis, github, community, resources, github_repos, documentation)
- **Last Updated**: 2025-09-20T12:03:04.787130
- **Scraping Success Rate**: 100% (12/12 resources successfully scraped)

## 🔗 Quick Links

- [Chia Main Documentation](https://docs.chia.net/)
- [Chialisp Documentation](https://chialisp.com/)
- [Developer Guides](https://docs.chia.net/dev-guides-home/)
- [RPC Reference](https://docs.chia.net/reference-client/rpc-reference/rpc/)
- [CLI Reference](https://docs.chia.net/reference-client/cli-reference/cli/)
- [Spacescan API](https://docs.spacescan.io/api/address/xch_balance/)
- [Mintgarden API](https://api.mintgarden.io/docs)
- [Dexie API](https://dexie.space/api)
- [Chia GitHub](https://github.com/chia-network)
- [Chia Discord](https://discord.com/invite/chia)
- [Spacescan Explorer](https://www.spacescan.io/)

## 🤝 Contributing

The Chia resources database is automatically maintained through the scraper. To add new resources:

1. Add the resource URL to the appropriate category in `chia_resource_scraper.py`
2. Run the scraper: `python chia_resource_scraper.py`
3. The new resource will be automatically scraped and added to the database

## 📄 License

This Chia resources database is provided for educational and development purposes. Please respect the terms of service of the original websites and APIs.
