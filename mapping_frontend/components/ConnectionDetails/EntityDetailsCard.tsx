// components/ConnectionDetails/EntityDetailsCard.tsx
import React from 'react';
import { EntityDetails } from '../../lib/types';

interface EntityDetailsCardProps {
  entityId: string;
  details?: EntityDetails;
  loading: boolean;
}

const EntityDetailsCard: React.FC<EntityDetailsCardProps> = ({ 
  entityId, 
  details, 
  loading 
}) => {
  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-sm p-4 h-full animate-pulse">
        <div className="h-5 bg-gray-200 rounded w-3/4 mb-3"></div>
        <div className="h-4 bg-gray-200 rounded w-full mb-2"></div>
        <div className="h-4 bg-gray-200 rounded w-5/6 mb-2"></div>
        <div className="h-4 bg-gray-200 rounded w-4/6 mb-4"></div>
        <div className="h-10 bg-gray-200 rounded w-full mb-3"></div>
        <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
        <div className="h-4 bg-gray-200 rounded w-1/2 mb-2"></div>
      </div>
    );
  }

  if (!details) {
    return (
      <div className="bg-white rounded-lg shadow-sm p-4 h-full">
        <p className="text-gray-500">No details available for this entity.</p>
      </div>
    );
  }

  const organization = details.organization?.[0];
  
  return (
    <div className="bg-white rounded-lg shadow-sm p-4 h-full">
      <h3 className="font-semibold text-lg mb-2">
        {organization?.name || `Entity ${entityId.substring(0, 8)}...`}
      </h3>
      
      {organization?.description && (
        <p className="text-gray-700 text-sm mb-3">
          {organization.description}
        </p>
      )}
      
      <div className="space-y-3">
        {/* Contact Information */}
        {(organization?.url || organization?.email || (details.phone && details.phone.length > 0)) && (
          <div>
            <h4 className="font-medium text-sm text-gray-900 mb-1">Contact Information</h4>
            <div className="pl-1 space-y-1 text-sm">
              {organization?.url && (
                <p className="flex items-start">
                  <svg className="w-4 h-4 mr-2 mt-0.5 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                  </svg>
                  <a href={organization.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline truncate max-w-xs">
                    {organization.url}
                  </a>
                </p>
              )}
              
              {organization?.email && (
                <p className="flex items-start">
                  <svg className="w-4 h-4 mr-2 mt-0.5 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                  <a href={`mailto:${organization.email}`} className="text-blue-600 hover:underline">
                    {organization.email}
                  </a>
                </p>
              )}
              
              {details.phone && details.phone.length > 0 && details.phone.slice(0, 2).map((phone, index) => (
                <p key={phone.id || index} className="flex items-start">
                  <svg className="w-4 h-4 mr-2 mt-0.5 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                  </svg>
                  <span>
                    {phone.number}
                    {phone.extension && ` ext. ${phone.extension}`}
                    {phone.type && ` (${phone.type})`}
                  </span>
                </p>
              ))}
              {details.phone && details.phone.length > 2 && (
                <p className="text-xs text-gray-500 pl-6">
                  + {details.phone.length - 2} more phone number(s)
                </p>
              )}
            </div>
          </div>
        )}
        
        {/* Address */}
        {details.address && details.address.length > 0 && (
          <div>
            <h4 className="font-medium text-sm text-gray-900 mb-1">Address</h4>
            <div className="pl-1 space-y-1 text-sm">
              {details.address.slice(0, 2).map((address, index) => (
                <p key={address.id || index} className="flex items-start">
                  <svg className="w-4 h-4 mr-2 mt-0.5 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  <span>
                    {address.address_1}
                    {address.address_2 && <>, {address.address_2}</>}
                    <br />
                    {address.city}, {address.state_province} {address.postal_code}
                    {address.country && address.country !== 'US' && `, ${address.country}`}
                  </span>
                </p>
              ))}
              {details.address.length > 2 && (
                <p className="text-xs text-gray-500 pl-6">
                  + {details.address.length - 2} more address(es)
                </p>
              )}
            </div>
          </div>
        )}
        
        {/* Services */}
        {details.service && details.service.length > 0 && (
          <div>
            <h4 className="font-medium text-sm text-gray-900 mb-1">Services ({details.service.length})</h4>
            <div className="pl-1 space-y-2 text-sm">
              {details.service.slice(0, 3).map((service, index) => (
                <div key={service.id || index}>
                  <p className="font-medium">{service.name}</p>
                  {service.description && (
                    <p className="text-gray-600 text-xs">
                      {service.description.length > 100 
                        ? service.description.substring(0, 100) + '...' 
                        : service.description}
                    </p>
                  )}
                </div>
              ))}
              {details.service.length > 3 && (
                <p className="text-xs text-blue-600">
                  + {details.service.length - 3} more service(s)
                </p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default EntityDetailsCard;