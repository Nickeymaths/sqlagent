import re

from config import EXTRACT_COLUMN_REGEX, EXTRACT_FOREIGN_KEY_REGEX, EXTRACT_PRIMARY_KEY_REGEX, EXTRACT_TABLE_REGEX


def clean_identifier(identifier):
    """Remove quotes and backticks from SQL identifiers"""
    return identifier.strip().strip('"`[]')

def parse_sql_schema_with_regex(schema):
        """Parse SQL schema using the provided regex patterns"""
        
        
        # Find all matches using provided regex patterns
        flags = re.MULTILINE | re.IGNORECASE
        schema = re.sub(r'insert into .+?;', '', schema, flags=flags)
        
        table_matches = list(re.finditer(EXTRACT_TABLE_REGEX, schema, flags=flags))
        column_matches = list(re.finditer(EXTRACT_COLUMN_REGEX, schema, flags=flags))
        primary_key_matches = list(re.finditer(EXTRACT_PRIMARY_KEY_REGEX, schema, flags=flags))
        foreign_key_matches = list(re.finditer(EXTRACT_FOREIGN_KEY_REGEX, schema, flags=flags))
        
        results = {
            'tables': [],
            'columns': [],
            'primary_keys': [],
            'foreign_keys': []
        }
        
        # Extract table names
        for match in table_matches:
            table_name = clean_identifier(match.group(1))
            results['tables'].append(table_name)
        
        # Create a mapping of table positions to associate columns with tables
        table_positions = [(match.start(), clean_identifier(match.group(1))) for match in table_matches]
        
        # Extract columns and associate with tables
        for match in column_matches:
            column_name = clean_identifier(match.group(1))
            column_type = match.group(2)
            
            # Find which table this column belongs to
            column_position = match.start()
            table_name = None
            for i, (table_pos, tname) in enumerate(table_positions):
                if column_position > table_pos:
                    table_name = tname
                    # Check if there's a next table after this position
                    if i + 1 < len(table_positions) and column_position > table_positions[i + 1][0]:
                        continue
                    else:
                        break
            
            if table_name:
                results['columns'].append({
                    'table': table_name,
                    'column': column_name,
                    'type': column_type
                })
        
        # Extract primary keys and associate with tables
        for match in primary_key_matches:
            pk_position = match.start()
            table_name = None
            
            # Find which table this primary key belongs to
            for i, (table_pos, tname) in enumerate(table_positions):
                if pk_position > table_pos:
                    table_name = tname
                    if i + 1 < len(table_positions) and pk_position > table_positions[i + 1][0]:
                        continue
                    else:
                        break
            
            if table_name:
                # Handle both inline primary key and separate primary key declaration
                if match.group(1):  # Inline primary key (column name)
                    pk_column = clean_identifier(match.group(1))
                    results['primary_keys'].append({
                        'table': table_name,
                        'column': pk_column
                    })
                elif match.group(2):  # Separate primary key declaration
                    pk_columns = match.group(2)
                    # Split by comma for composite primary keys
                    for col in pk_columns.split(','):
                        pk_column = clean_identifier(col.strip())
                        results['primary_keys'].append({
                            'table': table_name,
                            'column': pk_column
                        })
        
        # Extract foreign keys and associate with tables
        for match in foreign_key_matches:
            fk_position = match.start()
            table_name = None
            
            # Find which table this foreign key belongs to
            for i, (table_pos, tname) in enumerate(table_positions):
                if fk_position > table_pos:
                    table_name = tname
                    if i + 1 < len(table_positions) and fk_position > table_positions[i + 1][0]:
                        continue
                    else:
                        break
            
            if table_name:
                fk_column = clean_identifier(match.group(1))
                ref_table = clean_identifier(match.group(2))
                ref_column = clean_identifier(match.group(3))
                
                results['foreign_keys'].append({
                    'table': table_name,
                    'column': fk_column,
                    'references_table': ref_table,
                    'references_column': ref_column
                })
        
        return results

def _build_context(columns, pks, fks):
    # Group by table
    tables = {}
    for c in columns:
        tables.setdefault(c['table'], []).append(f"{c['column']}: {c['type']}")
    
    # Group PKs
    pk_dict = {}
    for p in pks:
        pk_dict.setdefault(p['table'], []).append(p['column'])
    
    # Build output
    result = f"There are tables such as {', '.join(sorted(tables.keys()))}\n"
    
    for table in sorted(tables.keys()):
        cols = ", ".join(tables[table])
        pk = pk_dict.get(table, [])
        pk_list = ','.join(pk)
        pk_str = f". {pk[0]} is primary key" if len(pk) == 1 else f'. ({pk_list}) is primary key' if pk else ""
        result += f"Table {table} contains columns such as {cols}{pk_str}\n"
    
    for fk in fks:
        result += f"The {fk['column']} of {fk['table']} is foreign key of {fk['references_column']} of {fk['references_table']}\n"
    
    return result.strip().lower()
