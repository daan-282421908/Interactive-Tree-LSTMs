#!/usr/bin/env python

# Implementation of BioNLP Shared Task evaluation.

# Copyright (c) 2010-2013 BioNLP Shared Task organizers
# This script is distributed under the open source MIT license:
# http://www.opensource.org/licenses/mit-license

import sys
import re
import os
import optparse

# allowed types for entities, events, etc. (task-specific settings)
given_types = set([
        "Organism",
        "Organism_subdivision",
        "Anatomical_system",
        "Organ",
        "Multi-tissue_structure",
        "Tissue",
        "Cell",
        "Cellular_component",
        "Developing_anatomical_structure",
        "Organism_substance",
        "Immaterial_anatomical_entity",
        "Pathological_formation",
        "Gene_or_gene_product",
        "Drug_or_compound",
        "Protein_domain_or_region"
        ])

entity_types = given_types | set([])

event_types  = set([
        "Development",
        "Blood_vessel_development",
        "Growth",
        "Death",
        "Breakdown",
        "Cell_proliferation",
        "Remodeling",
        "Reproduction",
        "Metabolism",
        "Synthesis",
        "Catabolism",
        "Gene_expression",
        "Transcription",
        "Phosphorylation",
        "Dephosphorylation",
        "Pathway",
        "Binding",
        "Localization",
        "Regulation",
        "Positive_regulation",
        "Negative_regulation",
        "Planned_process",
    ])

output_event_type_order = [
        "Development",
        "Blood_vessel_development",
        "Growth",
        "Death",
        "Breakdown",
        "Cell_proliferation",
        "Remodeling",
        "Reproduction",
        '=[ANATOMY-TOTAL]= ',
        "Metabolism",
        "Synthesis",
        "Catabolism",
        "Gene_expression",
        "Transcription",
        "Phosphorylation",
        "Dephosphorylation",
        "Pathway",
        '=[MOLECUL-TOTAL]= ',
        "Binding",
        "Localization",
        "Regulation",
        "Positive_regulation",
        "Negative_regulation",
        '=[GENERAL-TOTAL]= ',
        "Planned_process",
        ' ====[TOTAL]====  ',
        ]

subtotal_event_set = {
    '=[ANATOMY-TOTAL]= ' : [
        "Development",
        "Blood_vessel_development",
        "Growth",
        "Death",
        "Breakdown",
        "Cell_proliferation",
        "Remodeling",
        "Reproduction",
        ],

    '=[MOLECUL-TOTAL]= ' : [
        "Metabolism",
        "Synthesis",
        "Catabolism",
        "Gene_expression",
        "Transcription",
        "Phosphorylation",
        "Dephosphorylation",
        "Pathway",
        ],

    '=[GENERAL-TOTAL]= ' : [
        "Binding",
        "Localization",
        "Regulation",
        "Positive_regulation",
        "Negative_regulation",
        ],
    
    ' ====[TOTAL]====  ' : [
        "Development",
        "Blood_vessel_development",
        "Growth",
        "Death",
        "Breakdown",
        "Cell_proliferation",
        "Remodeling",
        "Reproduction",
        "Metabolism",
        "Synthesis",
        "Catabolism",
        "Gene_expression",
        "Transcription",
        "Phosphorylation",
        "Dephosphorylation",
        "Pathway",
        "Binding",
        "Localization",
        "Regulation",
        "Positive_regulation",
        "Negative_regulation",
        "Planned_process",
        ]
    }
