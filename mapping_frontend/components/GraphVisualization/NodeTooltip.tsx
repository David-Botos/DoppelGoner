// components/GraphVisualization/NodeTooltip.tsx
import React from 'react';
import { Entity, EntityDetails } from '../../lib/types';

interface NodeTooltipProps {
  node: Entity;
  details?: EntityDetails;
  position: { x: number; y: number };
  loading: boolean;
}

const NodeTooltip: React.FC<NodeTooltipProps> = ({ 
  node, 
  details, 
  position, 
  loading 
}) => {
  if (!node) return null;
  
  return (
    <div
      className="absolute z-10 pointer-events-auto"
      style={{
        left: position.x + 10,
        top: position.y + 10,
        maxWidth: "350px",
      }}
    >
      <div className="p-4 shadow-lg bg-white rounded-lg border border-gray-200">
        <h3 className="font-bold text-lg mb-2">{node.name}</h3>
        
        {loading ? (
          <div className="flex items-center space-x-2 py-2">
            <div className="w-4 h-4 rounded-full border-2 border-t-transparent border-blue-500 animate-spin"></div>
            <span className="text-sm text-gray-600">Loading details...</span>
          </div>
        ) : !details ? (
          <p className="text-sm text-gray-600">No details available</p>
        ) : (
          <>
            {details.organization[0]?.description && (
              <p className="text-sm text-gray-600 mb-3">
                {details.organization[0].description.length > 100
                  ? `${details.organization[0].description.substring(0, 100)}...`
                  : details.organization[0].description}
              </p>
            )}
            
            {details.organization[0]?.url && (
              <p className="text-sm mb-1 flex items-center">
                <svg className="w-4 h-4 mr-2 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                </svg>
                <a
                  href={details.organization[0].url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-500 hover:underline"
                >
                  {details.organization[0].url}
                </a>
              </p>
            )}
            
            {details.organization[0]?.email && (
              <p className="text-sm mb-1 flex items-center">
                <svg className="w-4 h-4 mr-2 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
                <a
                  href={`mailto:${details.organization[0].email}`}
                  className="text-blue-500 hover:underline"
                >
                  {details.organization[0].email}
                </a>
              </p>
            )}
            
            {details.phone && details.phone.length > 0 && (
              <p className="text-sm mb-1 flex items-center">
                <svg className="w-4 h-4 mr-2 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                </svg>
                {details.phone[0].number}
                {details.phone[0].extension && ` ext. ${details.phone[0].extension}`}
              </p>
            )}
            
            {details.address && details.address.length > 0 && (
              <p className="text-sm mt-2 flex items-start">
                <svg className="w-4 h-4 mr-2 mt-0.5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                <span>
                  {details.address[0].address_1}
                  {details.address[0].address_2 && `, ${details.address[0].address_2}`}
                  <br />
                  {details.address[0].city}, {details.address[0].state_province} {details.address[0].postal_code}
                </span>
              </p>
            )}
            
            {details.service && details.service.length > 0 && (
              <div className="mt-3">
                <p className="text-sm font-medium mb-1 flex items-center">
                  <svg className="w-4 h-4 mr-2 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                  </svg>
                  Services:
                </p>
                <ul className="text-xs text-gray-600 pl-6 list-disc">
                  {details.service.slice(0, 2).map((service) => (
                    <li key={service.id} className="mb-0.5">
                      {service.name}
                    </li>
                  ))}
                  {details.service.length > 2 && (
                    <li className="text-blue-500">
                      {details.service.length - 2} more services...
                    </li>
                  )}
                </ul>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default NodeTooltip;