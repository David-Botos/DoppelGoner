// app/api/entities/[entity_id]/details/route.ts
import { EntityDetails } from '@/lib/types';
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// Mock data for the MVP - in a real application, this would be fetched from a database
const mockEntityDetails: Record<string, EntityDetails> = {
  'entity1': {
    id: 'entity1',
    organization_name: 'Memorial Hospital',
    organization_url: 'https://memorial.org',
    phones: [
      { number: '555-111-2222', extension: null, type: 'voice' },
      { number: '555-111-3333', extension: '123', type: 'fax' }
    ],
    addresses: [
      { 
        address_1: '123 Hospital Ave', 
        address_2: 'Suite 100', 
        city: 'Metropolis', 
        state_province: 'CA', 
        postal_code: '90001', 
        country: 'US' 
      }
    ],
    locations: [
      { latitude: 34.0522, longitude: -118.2437 }
    ],
    services: [
      { id: 'service1', name: 'Emergency Care', short_description: '24/7 emergency medical services' },
      { id: 'service2', name: 'Outpatient Surgery', short_description: 'Scheduled surgical procedures' }
    ]
  },
  'entity2': {
    id: 'entity2',
    organization_name: 'Memorial Healthcare',
    organization_url: 'https://memorialhealth.org',
    phones: [
      { number: '555-222-3333', extension: null, type: 'voice' }
    ],
    addresses: [
      { 
        address_1: '456 Health Blvd', 
        city: 'Metropolis', 
        state_province: 'CA', 
        postal_code: '90002', 
        country: 'US' 
      }
    ],
    locations: [
      { latitude: 34.0523, longitude: -118.2438 }
    ],
    services: [
      { id: 'service3', name: 'Primary Care', short_description: 'General healthcare services' },
      { id: 'service4', name: 'Specialist Referrals', short_description: 'Connections to specialized care' }
    ]
  },
  'entity3': {
    id: 'entity3',
    organization_name: 'City Medical Center',
    organization_url: 'https://citymedical.org',
    phones: [
      { number: '555-333-4444', extension: null, type: 'voice' }
    ],
    addresses: [
      { 
        address_1: '789 Medical Dr', 
        city: 'Metropolis', 
        state_province: 'CA', 
        postal_code: '90003', 
        country: 'US' 
      }
    ],
    locations: [
      { latitude: 34.0524, longitude: -118.2439 }
    ],
    services: [
      { id: 'service5', name: 'Urgent Care', short_description: 'Non-emergency medical care' },
      { id: 'service6', name: 'Physical Therapy', short_description: 'Rehabilitation services' }
    ]
  }
};

// Pre-populate the mock data for all entities
for (let i = 4; i <= 12; i++) {
  mockEntityDetails[`entity${i}`] = {
    id: `entity${i}`,
    organization_name: `Organization ${i}`,
    organization_url: `https://org${i}.org`,
    phones: [
      { number: `555-${i}00-${i}000`, extension: null, type: 'voice' }
    ],
    addresses: [
      { 
        address_1: `${i}00 Main St`, 
        city: 'Metropolis', 
        state_province: 'CA', 
        postal_code: `9000${i}`, 
        country: 'US' 
      }
    ],
    locations: [
      { latitude: 34.0520 + i/1000, longitude: -118.2430 - i/1000 }
    ],
    services: [
      { id: `service${i}a`, name: `Service ${i}A`, short_description: `Description for service ${i}A` },
      { id: `service${i}b`, name: `Service ${i}B`, short_description: `Description for service ${i}B` }
    ]
  };
}

export async function GET(
  request: NextRequest,
  { params }: { params: { entity_id: string } }
) {
  const {entity_id} = await params;
  
  try {
    // In a real app, you would fetch from a database here with appropriate joins
    // e.g., const entityDetails = await db.query('SELECT ... FROM public.entity e JOIN public.organization o ON ... WHERE e.id = $1', [entity_id]);
    
    const details = mockEntityDetails[entity_id];
    
    if (!details) {
      return NextResponse.json(
        { error: `Entity with ID ${entity_id} not found` },
        { status: 404 }
      );
    }
    
    // Simulate network latency for lazy loading demonstration
    // In a real app, you would just return the response immediately
    await new Promise(resolve => setTimeout(resolve, 300));
    
    return NextResponse.json(details);
  } catch (error) {
    console.error(`Error fetching details for entity ${entity_id}:`, error);
    return NextResponse.json(
      { error: 'Failed to fetch entity details' },
      { status: 500 }
    );
  }
}