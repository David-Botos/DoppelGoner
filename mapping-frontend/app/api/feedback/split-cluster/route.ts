// app/api/feedback/split-cluster/route.ts
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { DecisionType } from '../../../../lib/types';

interface SplitClusterRequest {
  reviewer_id: string;
  entity_group_id: string;
  method_type: string;
  entity_id_1: string;
  entity_id_2: string;
  cluster_id: string;
  split_entities: string[];
}

export async function POST(request: NextRequest) {
  try {
    const feedback: SplitClusterRequest = await request.json();
    
    // Validate the request
    if (!feedback.reviewer_id || !feedback.entity_group_id || 
        !feedback.method_type || !feedback.entity_id_1 || 
        !feedback.entity_id_2 || !feedback.cluster_id || 
        !feedback.split_entities || !feedback.split_entities.length) {
      return NextResponse.json(
        { error: 'Missing required fields in split cluster request' },
        { status: 400 }
      );
    }
    
    // In a real app, this would implement a database transaction:
    // 1. Create human_review_decisions record with type SPLIT_CLUSTER
    // 2. Create human_review_method_feedback record with was_correct=false
    // 3. Generate a new cluster for the split entities
    // 4. Update group_cluster_id in entity_group records for the new cluster
    
    console.log('Split cluster feedback:', {
      reviewer_id: feedback.reviewer_id,
      cluster_id: feedback.cluster_id,
      entity_group_id: feedback.entity_group_id,
      decision_type: DecisionType.SPLIT_CLUSTER,
      method_type: feedback.method_type,
      entity_id_1: feedback.entity_id_1,
      entity_id_2: feedback.entity_id_2,
      split_entities: feedback.split_entities,
      was_correct: false
    });
    
    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Return a mock new cluster ID (in real app, this would be the actual new cluster ID)
    return NextResponse.json({
      success: true,
      new_cluster_id: 'new_cluster_' + Math.floor(Math.random() * 1000)
    });
  } catch (error) {
    console.error('Error processing split cluster feedback:', error);
    return NextResponse.json(
      { error: 'Failed to process split cluster feedback' },
      { status: 500 }
    );
  }
}