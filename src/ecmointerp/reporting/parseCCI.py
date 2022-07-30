"""
Parses from raw text copied from: https://cran.r-project.org/web/packages/comorbidity/vignettes/comorbidityscores.html
"""
import json

# Replaces hyphen with literal range of codes
def _range_expander(unexpanded: list):
    ret = list()
    for entry in unexpanded:
        if "-" in entry:
            start, end = entry.split("-")
            start = start.strip()
            end = end.strip()

            # Assume all codes could start with a letter
            def split_alpha_num(str_in):
                alpha_prefix = ""
                numeric_part = 0
                for idx, c in enumerate(str_in):
                    if c.isalpha() or c == "0":
                        alpha_prefix += c
                    else:
                        numeric_part = int(str_in[idx:])
                        break
                else:  # for F00 - F03
                    alpha_prefix = alpha_prefix[0:-1]

                return alpha_prefix, numeric_part

            start_alpha_prefix, start_numeric_part = split_alpha_num(start)
            end_alpha_prefix, end_numeric_part = split_alpha_num(end)

            # If codes are ICD-10, they should both have same prefix (if ICD-9, prefix will be empty string)
            # assert start_alpha_prefix == end_alpha_prefix

            # Assuming inclusive ranges here
            print(
                f"[*] Including range: {start_alpha_prefix}{start_numeric_part} -> {end_alpha_prefix}{end_numeric_part}"
            )
            ret += [
                start_alpha_prefix + str(idx)
                for idx in range(start_numeric_part, end_numeric_part + 1)
            ]

        else:
            ret += [entry]

    return ret


def get_labeled_comorbidity_codes():
    dropped_categories = ["Myocardial infarction", "Cerebrovascular disease"]

    startswith_codes_by_category = dict()
    match_codes_by_category = dict()

    with open("reporting/raw_cci_codes.txt", "r") as f:
        for line in f.readlines():
            if line == "":
                continue

            line = line.strip()
            category, raw_codes = line.split(":")

            if category in dropped_categories:
                continue

            # Codes with no '.x' are easy: just have to match them exactly
            match_codes = [
                c.replace(".", "").strip()
                for c in raw_codes.split(",")
                if ".x" not in c
            ]
            match_codes = _range_expander(match_codes)

            # Codes with 'x' will be matched with any code that starts with the digits before the '.x'
            startswith_codes = [
                c.replace(".x", "").replace(".", "").strip()
                for c in raw_codes.split(",")
                if ".x" in c
            ]
            startswith_codes = _range_expander(startswith_codes)

            if category not in match_codes_by_category:
                match_codes_by_category[category] = list()

            if category not in startswith_codes_by_category:
                startswith_codes_by_category[category] = list()

            match_codes_by_category[category] += match_codes
            startswith_codes_by_category[category] += startswith_codes

    return match_codes_by_category, startswith_codes_by_category


def get_comorbidity_codes():
    mcodes, swcodes = get_labeled_comorbidity_codes()
    unlabeled_mcodes = [code for sublist in mcodes.values() for code in sublist]
    unlabeled_swcodes = [code for sublist in swcodes.values() for code in sublist]

    # Implicitly drop any codes that show up in multiple categories
    return set(unlabeled_mcodes), set(unlabeled_swcodes)


def save_comorbidity_codes():
    mcodes, swcodes = get_labeled_comorbidity_codes()

    assert len(mcodes.keys()) == len(
        swcodes.keys()
    ), "Every cci category SHOULD have both match and startswith codes"

    output = dict()

    # Calculate points for each comorbidity: most +1
    points = dict()
    for key in mcodes.keys():
        points[key] = 1

        if key == "Moderate or severe liver disease":
            points[key] = 3
        elif key == "Diabetes without chronic complication":
            points[key] = 2
        elif key == "Hemiplegia or paraplegia":
            points[key] = 2
        elif key == "Renal disease":
            points[key] = 2
        elif key == "Metastatic solid tumour":
            points[key] = 6
        elif (
            key
            == "Any malignancy, including lymphoma and leukaemia, except malignant neoplasm of skin"
        ):
            points[key] = 2
        elif key == "AIDS/HIV":
            points[key] = 6

    # After the assert, we can just iterate through one set of keys
    for key in mcodes.keys():
        output[key] = {
            "Match Codes": mcodes[key],
            "Startswith Codes": swcodes[key],
            "Points": points[key],
        }

    with open("reporting/cci.json", "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    save_comorbidity_codes()
    # a, b = get_comorbidity_codes()
    # print(len(a))
    # print(len(b))
