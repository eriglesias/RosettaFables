import re
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

def parse_fable_file(file_path: Path) -> Dict[str, Any]:
    """
    Parse a markdown file containing multiple fables into a structured format.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract the collection title
    title_match = re.search(r'^# (.*?)$', content, re.MULTILINE)
    collection_title = title_match.group(1).strip() if title_match else "Aesop's Fables"

    # This approach directly extracts all fable entries by looking for <fable_id> tags
    # rather than relying on section headers
    fable_entries = re.findall(r'(?:<fable_id>.*?</moral>|<fable_id>.*?</body>)', content, re.DOTALL)

    # Group entries by fable_id
    fables_by_id = {}

    for entry in fable_entries:
        # Parse the entry
        fable_id_match = re.search(r'<fable_id>(.*?)</fable_id>', entry, re.DOTALL)
        if not fable_id_match:
            continue

        fable_id = fable_id_match.group(1).strip()

        # Extract other metadata
        title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
        language_match = re.search(r'<language>(.*?)(?:</language>|<language)', entry, re.DOTALL)
        source_match = re.search(r'<source>(.*?)</source>', entry, re.DOTALL)
        version_match = re.search(r'<version>(.*?)</version>', entry, re.DOTALL)
        body_match = re.search(r'<body>(.*?)</body>', entry, re.DOTALL)
        moral_match = re.search(r'<moral(?: type="(.*?)")?>([^<]*)</moral>', entry, re.DOTALL)

        # Extract language, handling all possible cases
        language_text = ""
        if language_match:
            language_text = language_match.group(1).strip()

        # Find section header if present
        section_header = ""
        header_match = re.search(r'### (.*?)$', entry, re.MULTILINE)
        if header_match:
            section_header = header_match.group(1).strip()

        # Infer language from header if not found in tag
        if not language_text:
            if "English" in section_header:
                language_text = "en"
            elif "Dutch" in section_header:
                language_text = "nl"
            elif "German" in section_header:
                language_text = "de"
            elif "Spanish" in section_header:
                language_text = "es"
            elif "Greek" in section_header:
                language_text = "grc"

        # Create version data
        version_data = {
            "version_name": section_header or f"Version {version_match.group(1).strip() if version_match else '1'}",
            "fable_id": fable_id,
            "title": title_match.group(1).strip() if title_match else "",
            "language": clean_language_code(language_text),
            "source": source_match.group(1).strip() if source_match else "",
            "version": version_match.group(1).strip() if version_match else "1",
            "body": body_match.group(1).strip() if body_match else ""
        }

        # Add moral with type information
        if moral_match:
            moral_type = moral_match.group(1) if moral_match.group(1) else "implicit"
            moral_text = moral_match.group(2).strip() if moral_match.group(2) else ""
            version_data["moral"] = {
                "type": moral_type,
                "text": moral_text
            }
        else:
            version_data["moral"] = {
                "type": "absent",
                "text": ""
            }

        # Add to fables_by_id
        if fable_id not in fables_by_id:
            fables_by_id[fable_id] = {
                "id": fable_id,
                "title": version_data.get("title", ""),
                "versions": []
            }

        fables_by_id[fable_id]["versions"].append(version_data)

    # Convert dictionary to list
    fables = list(fables_by_id.values())

    return {
        "title": collection_title,
        "fables": fables
    }

def clean_language_code(language_code: str) -> str:
    """
    Clean and standardize language codes.
    
    Args:
        language_code: Raw language code from the markdown
        
    Returns:
        Cleaned language code
    """
    if not language_code:
        return ""

    # Map language codes to standard ISO codes if needed
    language_map = {
        'dutch': 'nl',
        'german': 'de',
        'spanish': 'es',
        'english': 'en',
        'greek': 'grc',
        'ancient greek': 'grc'
    }

    return language_map.get(language_code.lower(), language_code)

def validate_fable(fable: Dict[str, Any]) -> List[str]:
    """
    Validate a fable structure and return a list of warnings.
    
    Args:
        fable: A parsed fable dictionary
        
    Returns:
        List of warning messages (empty if no warnings)
    """
    warnings = []

    if not fable.get("id"):
        warnings.append(f"Fable '{fable.get('title', 'Unknown')}' has no ID")

    if not fable.get("versions"):
        warnings.append(f"Fable {fable.get('id', 'Unknown')} has no language versions")

    for version in fable.get("versions", []):
        if not version.get("language"):
            warnings.append(f"Fable {fable.get('id', 'Unknown')} has a version with no language code")
        if not version.get("body"):
            warnings.append(f"Fable {fable.get('id', 'Unknown')} ({version.get('language', 'unknown')}) has an empty body")

    return warnings

def process_fables_directory(input_dir: Path, output_dir: Path) -> None:
    """
    Process all fable files in a directory and convert to JSON.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    # Statistics 
    total_files = 0
    total_fables = 0
    total_versions = 0
    languages = set()

    # Process all markdown files in the input directory
    for file_path in input_dir.glob("*.md"):
        total_files += 1
        try:
            # Parse the file
            data = parse_fable_file(file_path)

            # Validate fables
            warnings = []
            for fable in data["fables"]:
                warnings.extend(validate_fable(fable))

            if warnings:
                print(f"Warnings for {file_path.name}:")
                for warning in warnings:
                    print(f"  - {warning}")

            # Collect statistics
            total_fables += len(data["fables"])
            for fable in data["fables"]:
                total_versions += len(fable["versions"])
                for version in fable["versions"]:
                    if version.get("language"):
                        languages.add(version["language"])

            # Save collection to JSON
            collection_file = output_dir / f"{file_path.stem}.json"
            with open(collection_file, 'w', encoding='utf-8') as f:  # FIXED: use collection_file
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Organize by language
            by_language = {}
            for fable in data["fables"]:
                for version in fable["versions"]:
                    lang = version["language"]
                    if not lang:  # Skip versions with no language code
                        continue

                    if lang not in by_language:
                        by_language[lang] = []

                    # Create language-specific version
                    lang_version = {
                        "id": fable["id"],
                        "title": version["title"],
                        "source": version["source"],
                        "body": version["body"],
                        "moral": version["moral"]
                    }
                    by_language[lang].append(lang_version)

            # Save language-specific files
            for lang, fables in by_language.items():
                lang_file = output_dir / f"fables_{lang}.json"
                with open(lang_file, 'w', encoding='utf-8') as f:
                    json.dump(fables, f, ensure_ascii=False, indent=2)

            print(f"Successfully processed {file_path.name}")

        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Print summary
    print(f"\nSummary:")
    print(f"  Files processed: {total_files}")
    print(f"  Total fables: {total_fables}")
    print(f"  Total language versions: {total_versions}")
    print(f"  Languages found: {', '.join(sorted(languages))}")