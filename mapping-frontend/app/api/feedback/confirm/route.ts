// app/api/feedback/confirm/route.ts
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { FeedbackRequest, DecisionType } from '../../../../lib/types';

export async function POST(request: NextRequest) {
  try {
    const feedback: FeedbackRequest = await request.json();
    
    // Validate the request
    if (!feedback.reviewer_id || !feedback.entity_group_id || 
        !feedback.method_type || !feedback.entity_id_1 || !feedback.entity_id_2) {
      return NextResponse.json(
        { error: 'Missing required fields in feedback request' },
        { status: 400 }
      );
    }
    
    // In a real app, you would implement a database transaction:
    // 1. Create human_review_decisions record with type CONFIRM_PAIRWISE_MATCH
    // 2. Create human_review_method_feedback record with was_correct=true
    
    console.log('Confirmed link feedback:', {
      reviewer_id: feedback.reviewer_id,
      entity_group_id: feedback.entity_group_id,
      decision_type: DecisionType.CONFIRM_PAIRWISE_MATCH,
      method_type: feedback.method_type,
      entity_id_1: feedback.entity_id_1,
      entity_id_2: feedback.entity_id_2,
      was_correct: true
    });
    
    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 200));
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error processing confirm feedback:', error);
    return NextResponse.json(
      { error: 'Failed to process feedback' },
      { status: 500 }
    );
  }
}