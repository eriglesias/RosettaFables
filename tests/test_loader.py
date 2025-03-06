import sys
from pathlib import Path
import json

# Get the project root directory (two levels up from the test file)
project_root = Path(__file__).resolve().parent.parent

# Add the src directory to Python's path
sys.path.insert(0, str(project_root / "src"))

# The import should be from the package, NOT a relative import
from aesop_spacy.data.loader import parse_fable_file, process_fables_directory

def test_parser():
    """Test the fable parser with the initial_fables.md file."""
    # Find the fable file
    data_dir = project_root / "data" / "raw" / "fables"
    fable_path = data_dir / "initial_fables.md"

    if not fable_path.exists():
        print(f"Error: File not found at {fable_path}")
        return

    # Parse the file
    data = parse_fable_file(fable_path)

    # Print basic statistics
    print(f"Collection title: {data['title']}")
    print(f"Number of fables: {len(data['fables'])}")
    
    # Basic assertions to ensure parser is working
    assert data['title'], "Collection title should not be empty"
    assert len(data['fables']) > 0, "Should find at least one fable"
    
    # Print details of the first fable
    first_fable = data['fables'][0]
    print(f"\nFirst fable: {first_fable['title']}")
    print(f"ID: {first_fable['id']}")
    print(f"Number of versions: {len(first_fable['versions'])}")
    
    # Print languages of all versions
    print("\nLanguages found:")
    language_versions = {}  # Track versions by language
    for fable in data['fables']:
        for version in fable['versions']:
            lang = version.get('language', 'unknown')
            if lang not in language_versions:
                language_versions[lang] = []
            language_versions[lang].append(version)
            print(f"  - {lang}: {version.get('title', 'No title')}")
    
    # Print sample morals
    print("\nSample morals:")
    for fable in data['fables'][:2]:  # First two fables
        for version in fable['versions'][:2]:  # First two versions of each
            moral = version.get('moral', {})
            print(f"  - {version.get('language', 'unknown')}: {moral.get('text', 'No moral')} "
                  f"(Type: {moral.get('type', 'unknown')})")

    # Count unique languages
    languages = set()
    for fable in data['fables']:
        for version in fable['versions']:
            if 'language' in version and version['language']:
                languages.add(version['language'])

    print(f"Languages found: {', '.join(sorted(languages))}")

    # Count moral types
    moral_types = {}
    for fable in data['fables']:
        for version in fable['versions']:
            moral = version.get('moral', {})
            moral_type = moral.get('type', 'absent')
            moral_types[moral_type] = moral_types.get(moral_type, 0) + 1

    print("\nMoral types:")
    for moral_type, count in sorted(moral_types.items()):
        print(f"  {moral_type}: {count}")

    # Output file for inspection
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(exist_ok=True, parents=True)

    output_path = output_dir / "fables_test.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nProcessed data saved to {output_path}")

    # Process all fables and create language-specific files
    print("\nProcessing all fables and creating language-specific files...")
    process_fables_directory(data_dir, output_dir)
    
    # Check that language files were created
    print("\nChecking language files:")
    for lang in languages:
        lang_file = output_dir / f"fables_{lang}.json"
        if lang_file.exists():
            with open(lang_file, 'r', encoding='utf-8') as f:
                lang_data = json.load(f)
                print(f"  {lang}: {len(lang_data)} fables")
                
                # If this is Greek, show titles to debug
                if lang == 'grc':
                    print("  Greek fables found:")
                    for fable in lang_data:
                        print(f"    - {fable.get('title', 'No title')}")
        else:
            print(f"  WARNING: No file for language '{lang}'")
            if lang in language_versions:
                print(f"    (We found {len(language_versions[lang])} versions during parsing)")

    print("\nDone!")

if __name__ == "__main__":
    test_parser()