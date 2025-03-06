import re
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

def parse_fable_file(file_path: Path) -> Dict[str, Any]:
    """
    Parse a markdown file containing multiple fables into a structured format.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        Dictionary with collection title and list of fables
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract the collection title
    title_match = re.search(r'^# (.*?)$', content, re.MULTILINE)
    collection_title = title_match.group(1).strip() if title_match else "Aesop's Fables"

    # Split the content into sections by looking for level 2 headings
    sections = re.split(r'## ', content)[1:]  # Skip the header

    # Group language versions by fable_id
    fable_versions = []
    for section in sections:
        version = parse_version_section("## " + section)
        if version:
            fable_versions.append(version)

    # Group versions by fable_id
    fables_by_id = {}
    for version in fable_versions:
        fable_id = version.get('fable_id')
        if fable_id:
            if fable_id not in fables_by_id:
                fables_by_id[fable_id] = {
                    "id": fable_id,
                    "title": version.get('title', ''),
                    "versions": []
                }
            fables_by_id[fable_id]["versions"].append(version)

    # Convert dictionary to list
    fables = list(fables_by_id.values())

    return {
        "title": collection_title,
        "fables": fables
    }

def parse_version_section(section: str) -> Optional[Dict[str, Any]]:
    """
    Parse a section that contains a language version of a fable.
    
    Args:
        section: Text content of a section
        
    Returns:
        Dictionary with version data or None if parsing fails
    """
    # Extract the section title (language version name)
    title_match = re.search(r'## (.*?)$', section, re.MULTILINE)
    if not title_match:
        return None

    version_name = title_match.group(1).strip()

    # Extract components using regex
    fable_id_match = re.search(r'<fable_id>(.*?)</fable_id>', section, re.DOTALL)
    title_match = re.search(r'<title>(.*?)</title>', section, re.DOTALL)
    language_match = re.search(r'<language>(.*?)(?:</language>|<language)', section, re.DOTALL)
    source_match = re.search(r'<source>(.*?)</source>', section, re.DOTALL)
    version_match = re.search(r'<version>(.*?)</version>', section, re.DOTALL)
    body_match = re.search(r'<body>(.*?)</body>', section, re.DOTALL)

    # Handle potential malformed tags (missing closing tags)
    if body_match is None:
        body_match = re.search(r'<body>(.*)', section, re.DOTALL)

    # Extract moral with type attribute
    moral_match = re.search(r'<moral(?: type="(.*?)")?>([^<]*)</moral>', section, re.DOTALL)

    # Skip sections that appear to be headers without content
    if not fable_id_match and not body_match and len(section) < 50:
        return None

    # Infer language from section name if not found in tags
    language_text = ""
    if language_match:
        language_text = language_match.group(1).strip()
    elif "English Version" in version_name:
        language_text = "en"
    elif "Dutch Version" in version_name:
        language_text = "nl"
    elif "German Version" in version_name:
        language_text = "de"
    elif "Spanish Version" in version_name:
        language_text = "es"
    elif "Greek Version" in version_name or "Ancient Greek Version" in version_name:
        language_text = "grc"

    # Build result dictionary
    result = {
        "version_name": version_name,
        "fable_id": fable_id_match.group(1).strip() if fable_id_match else None,
        "title": title_match.group(1).strip() if title_match else "",
        "language": clean_language_code(language_text),
        "source": source_match.group(1).strip() if source_match else "",
        "version": version_match.group(1).strip() if version_match else "",
        "body": body_match.group(1).strip() if body_match else ""
    }

    # Add moral with type information
    if moral_match:
        moral_type = moral_match.group(1) if moral_match.group(1) else "implicit"
        moral_text = moral_match.group(2).strip() if moral_match.group(2) else ""
        result["moral"] = {
            "type": moral_type,
            "text": moral_text
        }
    else:
        result["moral"] = {
            "type": "absent",
            "text": ""
        }

    return result

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
    
    Args:
        input_dir: Directory containing markdown files
        output_dir: Directory where JSON files will be saved
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